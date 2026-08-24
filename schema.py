"""
schema.py — Text2Cypher 프롬프트용 스키마 텍스트 생성.

핵심 2가지:
 1) 살아있는 그래프를 introspect 한다. 빌드 스펙만 쓰면 '추출된 관계'가 스키마에 없어
    LLM이 그 엣지를 못 쓴다.
 2) rel_types 를 주면 **그 관계와 거기 등장하는 라벨만** 남긴 '유계 스키마'를 만든다.
    이것이 전략 D의 '논리적 서브그래프' 실체다. 전역 스키마를 통째로 주면 소형 모델이
    관계 이름을 지어내고 문법이 깨진다(실측).
"""

from __future__ import annotations

import re
from collections import Counter

KEY_PROPS = {
    "Product":       "product_id, product_name, line_id, unit_cost, list_price, safety_stock",
    "Supplier":      "supplier_id, supplier_name, country, risk_tier, lead_time_days, payment_terms",
    "Equipment":     "equipment_id, equipment_name, line_id",
    "Line":          "line_id, line_name",
    "Region":        "region_id, region_name",
    "Customer":      "customer_id, customer_name, region_id",
    "AnomalyEvent":  "anomaly_id, equipment_id, event_time, severity, anomaly_type, description",
    "DowntimeEvent": "downtime_id, equipment_id, anomaly_id, start_time, end_time, hours, affected_product_id, description",
    "ServiceTicket": "ticket_id, customer_id, product_id, opened_at, closed_at, priority, category, status, subject",
    "PurchaseOrder": "purchase_order_id, supplier_id, order_date, expected_date, status, incoterm",
    "Contract":      "contract_id, customer_id, start_date, end_date, service_level, annual_value",
    "Document":      "document_id, title, document_date, source, text",
    "Chunk":         "chunk_id, doc_id, seq, text",
}

# 관계 의미 주석(gloss). structured/document_link 는 아래 정적 사전(데이터딕셔너리, 중립적 FK 의미).
# extracted 는 하드코딩하지 않고 그래프의 evidence(근거 구절)에서 자동 추출한다(extracted_glosses).
# 목적: 모델이 이름만이 아니라 '의미'로 관계를 고르게 — 편향 지시가 아니라 정보 제공.
_STRUCTURED_GLOSS = {
    "ON_LINE":          "is assigned to this production line",
    "ON_EQUIPMENT":     "was recorded on this equipment",
    "LINKED_ANOMALY":   "this downtime is linked to this anomaly event",
    "AFFECTED_PRODUCT": "this downtime halted production of this product",
    "IN_REGION":        "this customer is located in this region",
    "FROM_CUSTOMER":    "this service ticket was filed by this customer",
    "ABOUT_PRODUCT":    "this service ticket is about this product",
    "FROM_SUPPLIER":    "this purchase order was placed with this supplier",
    "FOR_CUSTOMER":     "this contract belongs to this customer",
    "SUPPLIED_BY":      "this product is contractually supplied by this supplier",
    "ORDERS_PRODUCT":   "this purchase order line orders this product as routine procurement",
    "DESCRIBES":        "this document mentions this entity",
}


def extracted_glosses(graph) -> dict:
    """extracted 타입별 대표 evidence(근거 구절)를 gloss로. 최빈(동률이면 최단), 짧게 트림.
    하드코딩이 아니라 그래프에서 뽑으므로 아는 브리지뿐 아니라 모든 extracted 타입에 자동 적용."""
    rows = graph.run_read(
        "MATCH ()-[r]->() WHERE r.origin='extracted' AND r.evidence IS NOT NULL AND r.evidence <> '' "
        "RETURN type(r) AS t, r.evidence AS ev",
        validate=False)
    bytype: dict[str, list] = {}
    for row in rows:
        bytype.setdefault(row["t"], []).append(row["ev"])
    out = {}
    for t, evs in bytype.items():
        best = sorted(Counter(evs).items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        g = re.sub(r"\s+", " ", str(best)).strip()
        out[t] = (g[:100] + "…") if len(g) > 100 else g
    return out


def build_rel_glosses(graph=None, rel_catalog=None) -> dict:
    """관계타입 -> gloss 맵(전부 오프라인 수기, 깨끗). init 시 1회 계산해 캐시.
      - 브리지(extracted 정준): extract._ENDPOINT_SENSES 의 프로토타입(pair별 predicate 지정 때 쓴 서술).
      - structured: _STRUCTURED_GLOSS 정적 사전.
    정준 인벤토리에 없는 잡동사니 추출타입(SUPPLIES, CONCERNS 등)은 gloss 없음(노이즈 방지).
    (evidence 자동추출은 ID 박힘·sense-split 오염 문제로 폐기 — extracted_glosses 는 미사용.)"""
    gl = {}
    try:
        from extract import SENSE_GLOSS               # 지연 임포트(순환 방지)
        gl.update(SENSE_GLOSS)
    except Exception as e:
        print(f"  [warn] SENSE_GLOSS 로드 실패({type(e).__name__}) -> structured gloss 만 사용")
    for t, g in _STRUCTURED_GLOSS.items():
        gl.setdefault(t, g)
    return gl


CAVEATS = """Rules:
- Relationship directions above are exact. Copy them character-for-character.
- Every relationship has `origin`: 'structured' (from relational tables), 'document_link'
  (document -> entity), or 'extracted' (LLM-extracted from document text).
  All three are valid evidence; use whichever answers the question.
- 'extracted' relationships also have `confidence`, `source_doc`, `source_chunk_id`, `evidence`.
  Add `WHERE r.confidence >= {tau}` and RETURN `source_chunk_id` when you traverse one.
- Use ONLY relationship types listed above. Never invent a relationship type.
- Do not put a relationship type inside a node pattern, and never use spaces in a RETURN alias.
- Return ids plus the readable fields needed to answer."""


def relationship_catalog(graph) -> list[tuple[str, str, str, str, int]]:
    """실제 그래프의 (소스라벨, 관계, 타겟라벨, origin, 건수)."""
    rows = graph.run_read(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a)[0] AS a, type(r) AS t, labels(b)[0] AS b, "
        "       coalesce(r.origin,'unknown') AS o, count(*) AS c "
        "ORDER BY t",
        validate=False,
    )
    return [(r["a"], r["t"], r["b"], r["o"], r["c"]) for r in rows if r["a"] and r["b"]]


def build_schema_text(graph, tau: float = 0.5, rel_types: list[str] | None = None,
                      catalog: list | None = None, extra_labels: list[str] | None = None,
                      glosses: dict | None = None) -> str:
    """rel_types 가 주어지면 그 관계만 남긴 '유계 스키마'를 만든다(전략 D의 스코프)."""
    cat = catalog if catalog is not None else relationship_catalog(graph)

    if rel_types:
        allow = set(rel_types)
        cat = [x for x in cat if x[1] in allow]

    # 스코프에 등장하는 라벨만 (+ 시드 라벨)
    labels = set()
    for (a, t, b, o, c) in cat:
        labels.add(a); labels.add(b)
    for lb in (extra_labels or []):
        labels.add(lb)

    node_lines = [f"  ({lb}) props: {KEY_PROPS.get(lb, '')}" for lb in sorted(labels)]

    rel_lines = []
    for (a, t, b, o, c) in sorted(set(cat), key=lambda x: (x[3], x[1], x[0])):
        line = f"  ({a})-[:{t}]->({b})   origin='{o}'  [{c}]"
        g = (glosses or {}).get(t)
        if g:
            line += f"   // {g}"
        rel_lines.append(line)

    return ("Node labels and properties:\n" + "\n".join(node_lines)
            + "\n\nRelationships (direction is exact):\n" + "\n".join(rel_lines)
            + "\n\n" + CAVEATS.format(tau=tau))