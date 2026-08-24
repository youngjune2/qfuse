"""
graph.py — Neo4j 읽기 전용 클라이언트 (검색 단계 전용).

- 쓰기 키워드 차단(검색 경로에서 그래프 변경 불가)
- 자동 LIMIT 주입
- EntityCatalog: 실제 그래프에서 ID/이름을 인덱싱해 질문의 시드 링킹에 사용
- relationship_types(): 전역 관계타입 조회(프루닝 입력)
- fetch_chunks(): extracted 엣지의 source_chunk_id -> 원문 회수(provenance)
"""

from __future__ import annotations

import os
import re
from typing import Any

from neo4j import GraphDatabase
from neo4j.time import Date, DateTime

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
MAX_ROWS       = int(os.getenv("MAX_ROWS", "50"))

_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|"
    r"apoc\.(create|merge|refactor)|CALL\s+db\.(create|drop))\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\b\s+\d+", re.IGNORECASE)

# 이름 컬럼이 있는 라벨 (id + name 인덱싱)
NAME_COLS = {
    "Product":   "product_name",
    "Supplier":  "supplier_name",
    "Equipment": "equipment_name",
    "Line":      "line_name",
    "Region":    "region_name",
    "Customer":  "customer_name",
}
# 이름이 없는 라벨 (ID만 인덱싱; 질문에 AE001/DT001 같은 ID가 직접 등장)
ID_ONLY = {
    "AnomalyEvent":  "anomaly_id",
    "DowntimeEvent": "downtime_id",
    "ServiceTicket": "ticket_id",
    "PurchaseOrder": "purchase_order_id",
    "Contract":      "contract_id",
    "Document":      "document_id",
}
LABEL_PK = {
    "Product": "product_id", "Supplier": "supplier_id", "Equipment": "equipment_id",
    "Line": "line_id", "Region": "region_id", "Customer": "customer_id", **ID_ONLY,
}


class CypherSafetyError(Exception):
    pass


class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.database = database or NEO4J_DATABASE
        self.driver = GraphDatabase.driver(
            uri or NEO4J_URI,
            auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
        )
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    # --- 안전 검증 + 실행 ---
    def validate(self, cypher: str) -> str:
        if _WRITE_PATTERN.search(cypher):
            raise CypherSafetyError("쓰기/위험 절이 감지되어 실행을 거부했습니다.")
        if ";" in cypher.strip().rstrip(";"):
            raise CypherSafetyError("복수 구문(;)은 허용되지 않습니다.")
        if not _LIMIT_PATTERN.search(cypher):
            cypher = cypher.rstrip().rstrip(";") + f"\nLIMIT {MAX_ROWS}"
        return cypher

    def run_read(self, cypher: str, params: dict | None = None,
                 validate: bool = True) -> list[dict]:
        if validate:
            cypher = self.validate(cypher)
        with self.driver.session(database=self.database) as session:
            result = session.execute_read(
                lambda tx: [r.data() for r in tx.run(cypher, params or {})]
            )
        return [_normalize(row) for row in result]

    # --- 스키마/카탈로그 ---
    def relationship_types(self) -> list[str]:
        rows = self.run_read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t",
            validate=False)
        return sorted(r["t"] for r in rows)

    def node_labels(self) -> list[str]:
        rows = self.run_read(
            "CALL db.labels() YIELD label RETURN label AS l", validate=False)
        return sorted(r["l"] for r in rows)

    def observed_triples(self) -> list[tuple[str, str, str]]:
        """실제 그래프에 존재하는 (소스라벨, 관계, 타겟라벨) 조합.
        빌드 스펙에 없는 '추출된 관계'까지 포함되므로 스키마 프롬프트에 필수."""
        rows = self.run_read(
            "MATCH (a)-[r]->(b) "
            "RETURN DISTINCT labels(a)[0] AS a, type(r) AS r, labels(b)[0] AS b",
            validate=False)
        return [(x["a"], x["r"], x["b"]) for x in rows if x["a"] and x["b"]]

    def node_properties(self) -> dict[str, set]:
        """라벨별 실제 속성 목록. 생성된 Cypher의 속성명 검증에 사용."""
        out: dict[str, set] = {}
        try:
            rows = self.run_read(
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
                "RETURN nodeLabels AS labels, propertyName AS prop", validate=False)
            for r in rows:
                for lb in (r.get("labels") or []):
                    out.setdefault(lb, set()).add(r.get("prop"))
        except Exception:
            pass
        return out

    def load_entity_catalog(self) -> "EntityCatalog":
        catalog = EntityCatalog()
        present = set(self.node_labels())
        for label, name_col in NAME_COLS.items():
            if label not in present:
                continue
            pk = LABEL_PK[label]
            for r in self.run_read(
                f"MATCH (n:`{label}`) RETURN n.`{pk}` AS id, n.`{name_col}` AS name",
                validate=False,
            ):
                catalog.add(label, r["id"], r.get("name"))
        for label, pk in ID_ONLY.items():
            if label not in present:
                continue
            for r in self.run_read(
                f"MATCH (n:`{label}`) RETURN n.`{pk}` AS id", validate=False
            ):
                catalog.add(label, r["id"], None)
        return catalog

    # --- provenance ---
    def fetch_chunks(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        return self.run_read(
            "MATCH (c:`Chunk`) WHERE c.chunk_id IN $ids "
            "RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.text AS text",
            {"ids": list(chunk_ids)}, validate=False)


class EntityCatalog:
    """엔티티 링킹용 경량 카탈로그. ID 정확매칭 + 이름 부분매칭."""

    def __init__(self):
        self.by_id: dict[tuple[str, str], str | None] = {}
        self._name_index: dict[str, tuple[str, str]] = {}

    def add(self, label: str, id_: str, name: str | None):
        if id_ is None:
            return
        self.by_id[(label, str(id_))] = name
        if name and len(str(name)) >= 4:
            self._name_index[str(name).lower()] = (label, str(id_))

    def find_in_text(self, text: str) -> list[tuple[str, str]]:
        hits: list[tuple[str, str]] = []
        low = (text or "").lower()
        for (label, id_) in self.by_id:
            if re.search(rf"\b{re.escape(id_)}\b", text, re.IGNORECASE):
                hits.append((label, id_))
        for name, li in self._name_index.items():
            if name in low and li not in hits:
                hits.append(li)
        seen, out = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    def name_of(self, label: str, id_: str) -> str | None:
        return self.by_id.get((label, id_))


def _normalize(row: dict) -> dict:
    return {k: _norm_val(v) for k, v in row.items()}


def _norm_val(v: Any) -> Any:
    if isinstance(v, (Date, DateTime)):
        return v.iso_format()
    if isinstance(v, dict):
        return {k: _norm_val(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    return v