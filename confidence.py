"""
confidence.py — 추출 트리플의 '규칙 기반 confidence' (독립 모듈)
================================================================
왜 규칙 기반인가:
  LLM이 스스로 뱉는 0~1 신뢰도(self-report)는 교정(calibration)이 안 돼 "0.5"의 근거가 없다.
  대신 '검증 가능한 신호'들을 가중합해서, τ(승격 임계값)가 해석을 갖도록 한다.

세 신호 (전부 0~1):
  1) endpoint_resolution : 양끝 엔티티가 정형 노드로 얼마나 확실히 해소됐나
                           (ID 정확매칭=1.0, 이름매칭=0.5, 실패=0.0 의 양끝 평균)
  2) evidence_grounding  : '관계(predicate)의 근거 구절'이 원문 chunk에 실제로 있나 (1/0)
                           (엔티티 존재는 s1이 담당 -> 중복 카운트 제거, 관계 근거만 본다)
  3) corroboration       : 같은 (subject,predicate,object)가 여러 chunk/문서에서 반복됐나
                           (1 - 1/n_sources)

  confidence = w1*s1 + w2*s2 + w3*s3      (기본 w = 0.4 / 0.4 / 0.2)
  승격(promote)  <=>  confidence >= tau   (기본 tau = 0.5)

이 모듈은 의존성이 없다(표준 라이브러리만). 추출 파이프라인은 이 모듈을 import해서 쓰고,
가중치/τ/신호 정의를 바꾸고 싶으면 여기만 수정하면 된다.

기본값의 함의(가중치 조정 후):
  근거(grounding)와 중복(corroboration) 비중을 키웠으므로, 양끝이 실노드여도
  원문 근거가 부실하면 τ(0.5)를 넘기 어렵다.
  예) 양끝 ID매칭(s1=1.0)이지만 근거 1/3, 중복 없음: 0.4*1.0+0.4*0.333+0.2*0 = 0.53 (경계)
      근거가 더 부실하면 탈락. 즉 '실노드 사실'만으로는 통과하지 못한다.
  더 조이거나 풀려면 w_* 와 tau 를 여기서 조정. 최종 튜닝은 document_links 골든(conf=1.000) 대조.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# 설정 (여기만 고치면 됨)
# ---------------------------------------------------------------------------
@dataclass
class ConfidenceConfig:
    # 가중치 조정: 엔드포인트 비중을 낮추고 원문근거·중복을 키움.
    # (근거 없는 환각 트리플이 τ를 넘지 못하게. 이전 0.6/0.3/0.1 -> 0.4/0.4/0.2)
    w_endpoint: float = 0.4
    w_grounding: float = 0.4
    w_corroboration: float = 0.2
    tau: float = 0.5                 # 승격 임계값
    # 엔드포인트 해소 방식별 점수
    score_id_match: float = 1.0      # 카탈로그 ID 정확매칭
    score_name_match: float = 0.5    # 이름 부분매칭
    score_no_match: float = 0.0      # 해소 실패

    def normalized_weights(self) -> tuple[float, float, float]:
        s = self.w_endpoint + self.w_grounding + self.w_corroboration
        if s <= 0:
            return (1.0, 0.0, 0.0)
        return (self.w_endpoint / s, self.w_grounding / s, self.w_corroboration / s)


# ---------------------------------------------------------------------------
# 입력 트리플
# ---------------------------------------------------------------------------
@dataclass
class ExtractedTriple:
    subject_id: str | None          # 카탈로그로 해소된 ID (실패면 None)
    predicate: str                  # 정규화된 관계 (UPPER_SNAKE)
    object_id: str | None
    subject_match: str = "none"     # 'id' | 'name' | 'none'
    object_match: str = "none"
    subject_surface: str = ""       # 원문에서의 표기(예: 'AE001' 또는 이름) — grounding 확인용
    object_surface: str = ""
    predicate_cue: str = ""         # 원문 근거 구절(예: 'supplied by') — grounding 확인용
    source_chunk_id: str = ""
    source_doc_id: str = ""
    chunk_text: str = ""            # 근거 chunk 원문


@dataclass
class ScoreBreakdown:
    triple: ExtractedTriple
    s_endpoint: float
    s_grounding: float
    s_corroboration: float
    confidence: float
    n_sources: int
    promoted: bool
    def as_dict(self) -> dict:
        t = self.triple
        return {
            "subject_id": t.subject_id, "predicate": t.predicate, "object_id": t.object_id,
            "s_endpoint": round(self.s_endpoint, 3),
            "s_grounding": round(self.s_grounding, 3),
            "s_corroboration": round(self.s_corroboration, 3),
            "confidence": round(self.confidence, 3),
            "n_sources": self.n_sources, "promoted": self.promoted,
            "source_chunk_id": t.source_chunk_id,
        }


# ---------------------------------------------------------------------------
# 개별 신호
# ---------------------------------------------------------------------------
def endpoint_score(t: ExtractedTriple, cfg: ConfidenceConfig) -> float:
    m = {"id": cfg.score_id_match, "name": cfg.score_name_match, "none": cfg.score_no_match}
    return (m.get(t.subject_match, 0.0) + m.get(t.object_match, 0.0)) / 2.0


def grounding_score(t: ExtractedTriple) -> float:
    """관계(predicate)의 근거 구절이 chunk 원문에 실제로 있는가 (1.0/0.0).

    주의: subject/object가 원문에 있는지는 엔드포인트 해소(s1)가 이미 담당한다.
    또한 추출을 'chunk 내 후보 엔티티끼리'로 제한하므로 엔티티는 항상 원문에 존재한다.
    따라서 s2는 '관계 자체가 텍스트에 진술됐는지'만 본다(중복 카운트 제거).
    predicate_cue(모델이 제시한 근거 span)가 원문에 없으면 = 관계가 지어내진 것으로 간주.
    """
    hay = re.sub(r"\s+", " ", (t.chunk_text or "").lower())
    cue = re.sub(r"\s+", " ", (t.predicate_cue or "").lower()).strip()
    if not hay or not cue:
        return 0.0
    return 1.0 if cue in hay else 0.0


def corroboration_score(n_sources: int) -> float:
    """서로 다른 출처(chunk) 개수 n에 대해 1 - 1/n. (1개=0.0, 2개=0.5, 3개=0.667 ...)"""
    if n_sources <= 1:
        return 0.0
    return 1.0 - (1.0 / n_sources)


# ---------------------------------------------------------------------------
# 스코어링
# ---------------------------------------------------------------------------
def _triple_key(t: ExtractedTriple) -> tuple:
    return (t.subject_id, t.predicate, t.object_id)


def score_one(t: ExtractedTriple, n_sources: int, cfg: ConfidenceConfig) -> ScoreBreakdown:
    s1 = endpoint_score(t, cfg)
    s2 = grounding_score(t)
    s3 = corroboration_score(n_sources)
    w1, w2, w3 = cfg.normalized_weights()
    conf = w1 * s1 + w2 * s2 + w3 * s3
    return ScoreBreakdown(
        triple=t, s_endpoint=s1, s_grounding=s2, s_corroboration=s3,
        confidence=conf, n_sources=n_sources, promoted=conf >= cfg.tau,
    )


def score_all(triples: list[ExtractedTriple], cfg: ConfidenceConfig | None = None) -> list[ScoreBreakdown]:
    """전체 추출 트리플을 스코어링. corroboration은 같은 (s,p,o)의 서로 다른 출처 chunk 수로 계산."""
    cfg = cfg or ConfidenceConfig()
    sources: dict[tuple, set] = defaultdict(set)
    for t in triples:
        sources[_triple_key(t)].add(t.source_chunk_id or id(t))
    return [score_one(t, len(sources[_triple_key(t)]), cfg) for t in triples]


def promoted_only(breakdowns: list[ScoreBreakdown]) -> list[ScoreBreakdown]:
    return [b for b in breakdowns if b.promoted]


# ---------------------------------------------------------------------------
# 자기 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chunk = ("Root-cause analysis on EQ001 (Checkweigher 001) traced the failure to a "
             "defective precision bearing supplied by Harvest Consumer Goods 31 (SUP031).")

    # 1) 좋은 트리플: 양끝 ID 정확매칭 + 근거 원문에 존재
    good = ExtractedTriple(
        subject_id="AE001", predicate="ATTRIBUTED_TO", object_id="SUP031",
        subject_match="id", object_match="id",
        subject_surface="AE001", object_surface="SUP031", predicate_cue="supplied by",
        source_chunk_id="DOC00258#c2", source_doc_id="DOC00258", chunk_text=chunk)

    # 2) 약한 트리플: object가 이름매칭 + predicate 근거가 원문에 없음(hallucination 냄새)
    weak = ExtractedTriple(
        subject_id="AE001", predicate="CAUSED_FIRE", object_id="SUP031",
        subject_match="id", object_match="name",
        subject_surface="AE001", object_surface="Harvest Consumer Goods 31",
        predicate_cue="started a fire", source_chunk_id="DOC00258#c2",
        source_doc_id="DOC00258", chunk_text=chunk)

    # 3) good과 동일 트리플이 다른 문서에서도 → corroboration 상승
    good2 = ExtractedTriple(
        subject_id="AE001", predicate="ATTRIBUTED_TO", object_id="SUP031",
        subject_match="id", object_match="id",
        subject_surface="AE001", object_surface="SUP031", predicate_cue="supplied by",
        source_chunk_id="DOC00259#c1", source_doc_id="DOC00259", chunk_text=chunk)

    import json
    cfg = ConfidenceConfig()
    print(f"cfg: w={cfg.w_endpoint}/{cfg.w_grounding}/{cfg.w_corroboration}  tau={cfg.tau}\n")
    for b in score_all([good, weak, good2], cfg):
        print(json.dumps(b.as_dict(), ensure_ascii=False))
    print("\n승격된 것만:", [b.triple.predicate for b in promoted_only(score_all([good, weak, good2], cfg))])
