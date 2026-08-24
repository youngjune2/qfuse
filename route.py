"""
route.py — baseline #1 의 라우터 코어(rule-driven query routing).

대조 프레임: "Route(질문마다 소스 택1) vs Unify(하나의 KG로 융합)". 우리 Strategy D 는
정형/비정형을 한 그래프로 통합해 Text2Cypher로 join 하는 쪽이고, 이 베이스라인은 통합하지
않고 질문을 보고 어느 소스로 보낼지 라우팅만 한다.

방식은 'Learning to Route: Rule-Driven Agent for Hybrid-Source RAG'(Chen et al., WWW'26,
arXiv:2510.02388)의 핵심(A_ROUTING)을 따른다:
  - 명시적 규칙집합 R 을 LLM 이 질문에 적용해 경로별 가산 점수 S_p(q)=A_ROUTING(q,p,R) 산출
  - p_q = argmax_p S_p(q)  (동점은 우선순위로 tie-break)
경로는 논문과 동일한 4-way: DB(정형 Text2Cypher) / Doc(문서 벡터) / Hybrid(둘 다) / LLM(검색 없음).

우리가 의도적으로 생략한 것(범위 축소):
  - A_RULE(규칙 자가개선 오프라인 루프): 100문항 규모에서 효과가 작고 비용만 큼 -> 고정 규칙집합
    버전(논문의 rule-agent ablation)에 해당. 규칙은 아래 DOMAIN_RULES 로 고정.
  - meta-cache(질의 임베딩 캐시): latency 최적화(직교) -> 라우팅 품질만 격리하려고 끈다.
"""

from __future__ import annotations

from llm import LLMClient

PATHS = ["DB", "Doc", "Hybrid", "LLM"]

# 동점 시 우선순위(앞이 우선). 증거를 더 많이 주는 Hybrid 를 앞에 둬 베이스라인에 유리하게
# (강한 baseline 이 그래도 지면 결과가 더 설득력 있음). LLM(무검색)은 최후.
TIE_PRIORITY = ["Hybrid", "DB", "Doc", "LLM"]

# 논문의 가산 규칙(숫자->DB, how/why->Doc, 정의->LLM, 사실+설명->Hybrid)을 우리 도메인에 맞춤.
# 정형 소스 = 관계형 테이블(PO/구매주문, product, supplier, equipment, downtime, anomaly,
#   contract, ticket 등 ID/수량/날짜/상태 필드). 비정형 소스 = 노트 문서(maintenance log,
#   ops incident note, procurement note, CRM case, contract note, quality review).
DOMAIN_RULES = """You route each question to ONE retrieval path by scoring the four paths with
these additive rules. Start every path at 0 and apply every rule that fires (scores add up).

Paths:
- DB    : Text2Cypher over the STRUCTURED relational tables only (purchase orders, products,
          suppliers, equipment, downtime, anomalies, contracts, tickets — their ids, quantities,
          dates, statuses, and the foreign-key links among them).
- Doc   : semantic (vector) search over the UNSTRUCTURED note documents only (maintenance logs,
          operations incident notes, procurement notes, CRM cases, contract notes, quality reviews).
- Hybrid: run BOTH the DB path and the Doc path and use both results.
- LLM   : answer directly from the model with NO retrieval (general definitions / common knowledge).

Rules:
1. If the question asks for a specific record, id, count, quantity, date, status, or a link that
   lives in the relational tables (e.g. "which supplier supplies product X", "how many POs",
   "what is the status of downtime DT..."), DB += 3.
2. If the question asks about the narrative CONTENT of a note — what a log/incident/CRM/contract
   note states, a described root cause, a "why did ..." or "how was ..." explanation that would be
   written in prose, or quotes/mentions from documents, Doc += 3.
3. If answering requires CONNECTING a structured record to something stated in a document — i.e. it
   names a concrete entity id/record AND asks about a document-described fact about it, or it must
   bridge a table fact to a note (root cause, attribution, an event tied to an equipment/PO/product),
   Hybrid += 3.
4. If the question is a general definition or common-sense question needing no company data, LLM += 3.
5. If the question mentions BOTH a structured entity (an id like PO..., PRD..., DT..., a supplier)
   AND a document concept (incident, note, root cause, review, complaint), Hybrid += 2.
6. If it is a pure yes/no membership or lookup fully answerable from one table, DB += 1.

Return integer scores for all four paths and a one-line reasoning."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "scores": {
            "type": "object",
            "properties": {p: {"type": "integer"} for p in PATHS},
            "required": PATHS,
        },
    },
    "required": ["scores"],
}


class RuleRouter:
    """A_ROUTING: 고정 규칙집합을 LLM 으로 적용해 경로별 점수 -> argmax(동점 tie-break)."""

    def __init__(self, llm: LLMClient, rules: str = DOMAIN_RULES,
                 tie_priority: list[str] | None = None):
        self.llm = llm
        self.rules = rules
        self.tie_priority = tie_priority or TIE_PRIORITY

    def route(self, question: str) -> dict:
        """질문 -> {'path','scores','reasoning'}. 실패 시 Hybrid 로 안전 폴백(증거 최대)."""
        out = self.llm.chat_json(self.rules, f"Question: {question}", _SCHEMA, max_tokens=300)
        raw = out.get("scores") or {}
        scores = {p: _as_int(raw.get(p)) for p in PATHS}
        path = self._argmax(scores)
        return {"path": path, "scores": scores, "reasoning": out.get("reasoning", "")}

    def _argmax(self, scores: dict) -> str:
        best = max(scores.values()) if scores else 0
        top = [p for p in PATHS if scores.get(p, 0) == best]
        if len(top) == 1:
            return top[0]
        for p in self.tie_priority:            # 동점 -> 우선순위 순
            if p in top:
                return p
        return top[0]


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
