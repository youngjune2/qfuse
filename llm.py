"""
llm.py — vLLM(OpenAI 호환) 클라이언트 공용 모듈.
추출(run_extract.py)과 검색(retrieve_d.py)이 함께 사용한다.

vLLM 최신 버전은 extra_body의 guided_json 을 무시한다(경고만 뜸).
대신 response_format={"type":"json_schema", ...} 를 쓴다. 서버가 이것도 거부하면
자동으로 '프롬프트만' 모드로 폴백하고, 견고한 파서가 결과를 수습한다.
"""

from __future__ import annotations

import os
import json
import re
from typing import Any

from openai import OpenAI

# --- 설정 (env로 덮어쓰기 가능) ---
# 추론 모델은 VLLM_MODEL 로 지정한다(추출 triple/llm_sense + retrieval Text2Cypher + synthesis 공용).
# 기본값은 Text2Cypher 특화 파인튜닝 모델(Gemma-3-27B). vLLM을 --served-model-name 없이 띄웠으면
# 서빙 이름이 곧 HF 경로라 아래 기본값이 맞고, 커스텀 이름으로 띄웠으면 VLLM_MODEL 로 덮어쓴다.
#   예) VLLM_MODEL=qwen3-coder            (이전 Qwen3-Coder-30B)
#       VLLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
VLLM_BASE_URL   = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY    = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL      = os.getenv("VLLM_MODEL", "t2c-gemma3-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# 구조화 출력(json_schema) 사용 여부. 0이면 프롬프트만으로 JSON 유도.
USE_STRUCTURED  = os.getenv("USE_STRUCTURED_OUTPUT", "1") not in ("0", "false", "False")

# 임베딩 엔드포인트(별도 vLLM 인스턴스). predicate 정규화(방법 2) 전용.
# 예: vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001
EMB_BASE_URL = os.getenv("EMB_BASE_URL", "http://localhost:8001/v1")
EMB_API_KEY  = os.getenv("EMB_API_KEY", "EMPTY")
EMB_MODEL    = os.getenv("EMB_MODEL", "qwen3-embedding")
  
def safe_json(raw: str, list_key: str | None = None) -> dict:
    """모델 출력 -> dict. 코드펜스/맨배열/잡음에 견디게.

    list_key 를 주면(예: "triples") 맨 배열로 온 응답을 {list_key: [...]} 로 감싼다.
    파싱 실패 시 조용히 넘어가지 않고 경고를 남긴다(원인 추적용).
    """
    txt = (raw or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt).strip()

    def _wrap(obj: Any):
        if isinstance(obj, list):
            return {list_key: obj} if list_key else {"items": obj}
        if isinstance(obj, dict):
            if list_key and list_key in obj:
                return obj
            if not list_key:
                return obj
            # 다른 키에 배열을 담아 준 경우 흡수
            for v in obj.values():
                if isinstance(v, list):
                    return {list_key: v}
            return obj
        return None

    try:
        w = _wrap(json.loads(txt))
        if w is not None:
            return w
    except json.JSONDecodeError:
        pass

    for op, cl in (("[", "]"), ("{", "}")):
        s, e = txt.find(op), txt.rfind(cl)
        if s != -1 and e > s:
            try:
                w = _wrap(json.loads(txt[s:e + 1]))
                if w is not None:
                    return w
            except json.JSONDecodeError:
                continue

    print(f"  [warn] JSON 파싱 실패: {txt[:150]!r}")
    return {list_key: []} if list_key else {}


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.model = model or VLLM_MODEL
        self.client = OpenAI(base_url=base_url or VLLM_BASE_URL,
                             api_key=api_key or VLLM_API_KEY)
        self._structured_ok = USE_STRUCTURED     # 실패하면 False로 내려 재시도 안 함
        # Gemma 계열은 chat template에 system role이 없어 vLLM이 거부한다.
        # 이 경우 system을 user 앞에 합쳐서 보낸다. VLLM_NO_SYSTEM_ROLE 로 강제 지정 가능.
        forced = os.getenv("VLLM_NO_SYSTEM_ROLE")
        if forced is not None:
            self._no_system_role = forced not in ("0", "false", "False", "")
        else:
            self._no_system_role = "gemma" in self.model.lower()

    def _messages(self, system: str, user: str) -> list[dict]:
        if self._no_system_role:
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, system: str, user: str, schema: dict,
                  list_key: str | None = None, max_tokens: int | None = None) -> dict:
        """구조화 출력 시도 -> 실패 시 프롬프트-only 폴백. 결과는 견고한 파서로 정리."""
        kwargs: dict[str, Any] = {}
        if self._structured_ok:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema},
            }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, user),
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS,
                **kwargs,
            )
        except Exception as e:
            if self._structured_ok:
                print(f"  [warn] 구조화 출력 미지원({type(e).__name__}) -> 프롬프트 모드로 폴백")
                self._structured_ok = False
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(system, user),
                    temperature=LLM_TEMPERATURE,
                    max_tokens=max_tokens or LLM_MAX_TOKENS,
                )
            else:
                raise
        return safe_json((resp.choices[0].message.content or "").strip(), list_key)


class EmbeddingClient:
    """별도 vLLM 임베딩 엔드포인트(OpenAI 호환 /v1/embeddings) 클라이언트.
    predicate 정규화(방법 2: 의미군집) 전용. 채팅 모델과 다른 인스턴스를 가리킨다."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.model = model or EMB_MODEL
        self.client = OpenAI(base_url=base_url or EMB_BASE_URL,
                             api_key=api_key or EMB_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # 서버가 순서를 보장하지만 방어적으로 index 정렬
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]
