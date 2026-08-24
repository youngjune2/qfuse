"""
retrieve_lightrag.py — LightRAG 스타일 검색을 우리 fused 그래프 위에서.

원논문 메커니즘(충실 재현):
  1) 질문에서 **dual-level 키워드** 추출:
       - low-level  : 구체 엔티티/개체(제품·공급사·설비·ID 등)
       - high-level : 상위 주제/관계 개념(root cause, downtime, procurement 등)
  2) low-level 키워드 -> 엔티티 노드 매칭(임베딩 최근접).
     high-level 키워드 -> 관계(엣지) 매칭(관계타입 설명 임베딩 최근접).
  3) 매칭된 엔티티의 1-hop 이웃 관계 + 매칭 관계타입의 엣지를 모으고,
     그 노드/엣지에 연결된 텍스트 청크를 함께 수집(dedup).
  4) 엔티티표 + 관계표 + 청크를 한 컨텍스트로 합성.

공정성: HippoRAG 어댑터와 동일 — 같은 살아있는 B2 그래프, 정형 속성 미차단, 같은 LLM/규칙/채점.
"""

from __future__ import annotations

from graph import EntityCatalog
from llm import LLMClient
from retrieve_graphrag import GraphData, EntityEmbeddingIndex

_KW_SCHEMA = {
    "type": "object",
    "properties": {
        "low_level": {"type": "array", "items": {"type": "string"}},
        "high_level": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["low_level", "high_level"],
}
_KW_SYSTEM = (
    "Extract keywords from the question at two levels for retrieval over a manufacturing/"
    "supply-chain knowledge graph.\n"
    "- low_level: concrete entities/specifics (ids like PRD001/SUP005/DT0007, product/supplier/"
    "equipment names, specific fields).\n"
    "- high_level: overarching themes / relation concepts (e.g. 'root cause', 'downtime', "
    "'procurement', 'supplier attribution', 'contract', 'complaint').\n"
    "Return JSON {\"low_level\": [...], \"high_level\": [...]}."
)


class LightRAGRetriever:
    strategy_name = "LightRAG"

    def __init__(self, gd: GraphData, llm: LLMClient, catalog: EntityCatalog,
                 emb_index: EntityEmbeddingIndex, k_low: int = 5, k_type: int = 3,
                 r_max: int = 30, ent_max: int = 30, chunk_max: int = 8,
                 link_tau: float = 0.5, link_margin: float = 0.05, debug: bool = False):
        self.gd = gd
        self.llm = llm
        self.catalog = catalog
        self.emb_index = emb_index
        self.k_low = k_low
        self.k_type = k_type
        self.r_max = r_max
        self.ent_max = ent_max
        self.chunk_max = chunk_max
        # low-keyword -> 엔티티 링킹 게이트(HippoRAG 와 동일 원칙). generic 키워드가 임의
        # 노이즈 엔티티를 주입해 1-hop 예산을 오염시키는 것을 막는다. exact seed 는 항상 유지.
        self.link_tau = link_tau
        self.link_margin = link_margin
        self.debug = debug

    def _keywords(self, question: str) -> tuple[list[str], list[str]]:
        try:
            out = self.llm.chat_json(_KW_SYSTEM, f"Question: {question}", _KW_SCHEMA,
                                     max_tokens=250)
            low = [k for k in (out.get("low_level") or []) if isinstance(k, str) and k.strip()]
            high = [k for k in (out.get("high_level") or []) if isinstance(k, str) and k.strip()]
            return low, high
        except Exception as e:
            if self.debug:
                print(f"  [lightrag] 키워드 추출 실패({type(e).__name__})")
            return [], []

    def retrieve(self, question: str) -> dict:
        low, high = self._keywords(question)

        # low-level -> 엔티티(정확 시드 ∪ 게이트된 임베딩 매칭)
        e0: list[str] = list(self.gd.link_seeds(question, self.catalog))
        for kw in low:
            m = self.emb_index.search_entities_gated(kw, k=self.k_low,
                                                     tau=self.link_tau, margin=self.link_margin)
            if self.debug:
                ids = [self.gd.nodes[e]["id"] for e in m]
                print(f"  [lightrag] low {kw!r} -> {ids or 'SKIP(ambiguous/generic)'}")
            e0.extend(m)
        e0 = list(dict.fromkeys(e0))
        e0_set = set(e0)

        # high-level -> 관계타입
        high_types: set[str] = set()
        for kw in high:
            high_types.update(self.emb_index.search_types(kw, k=self.k_type))

        # 매칭 엔티티의 1-hop 관계 수집 + 우선순위(high-타입 우대 -> origin/confidence)
        cand_idx = {i for eid in e0 for i in self.gd.incident.get(eid, ())}

        def _score(i: int) -> float:
            e = self.gd.edges[i]
            s = 2.0 if e["type"] in high_types else 0.0
            if e["origin"] in ("structured", "document_link"):
                s += 1.0
            elif e["cf"] is not None:
                s += float(e["cf"])
            return s

        rel_ids = sorted(cand_idx, key=_score, reverse=True)[:self.r_max]
        rels = [self.gd.edges[i] for i in rel_ids]

        # 이웃까지 엔티티 확장
        ent = list(e0)
        for e in rels:
            ent.append(e["s"])
            ent.append(e["t"])
        ent = [x for x in dict.fromkeys(ent) if self.gd.nodes[x]["label"] != "Document"][:self.ent_max]
        ent_set = set(ent)

        # 청크: E 노드가 많이 등장하는 청크 우선
        cand_chunks: set[str] = set()
        for eid in ent_set:
            cand_chunks.update(self.gd.node_chunks.get(eid, ()))
        for e in rels:
            if e["sc"] and e["sc"] in self.gd.chunks:
                cand_chunks.add(e["sc"])
        chunk_scored = sorted(
            ((len(self.gd.chunk_members.get(c, set()) & ent_set), c) for c in cand_chunks),
            reverse=True)
        chunks = [c for _, c in chunk_scored][:self.chunk_max]

        ent_ctx = "\n".join(f"  {self.gd.entity_text(e)}" for e in ent)
        rel_ctx = "\n".join(f"  {self.gd.edge_text(e)}" for e in rels)
        psg_ctx = "\n".join(f"  {self.gd.chunk_text(c)}" for c in chunks)

        return {
            "ent_ctx": ent_ctx, "rel_ctx": rel_ctx, "psg_ctx": psg_ctx,
            "seeds": [self.gd.nodes[e]["id"] for e in e0 if e in self.gd.nodes],
            "low_keywords": low, "high_keywords": high,
            "high_types": sorted(high_types),
            "passages": chunks,
            "entities": [self.gd.nodes[e]["id"] for e in ent],
            "meta": {"n_low": len(low), "n_high": len(high), "n_entities": len(ent),
                     "n_rels": len(rels), "n_passages": len(chunks)},
        }
