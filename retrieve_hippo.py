"""
retrieve_hippo.py — HippoRAG(NeurIPS'24) 스타일 검색을 우리 fused 그래프 위에서.

원논문 메커니즘(충실 재현):
  1) 질문 NER 로 named entity 추출.
  2) 그 엔티티를 KG 노드에 링킹(정확 ID/이름 매칭 + 선택적 임베딩 최근접).
  3) 링크된 노드를 seed 로 **Personalized PageRank** 를 그래프 전체에 전파.
  4) 각 passage(=문서 Chunk)를 '그 안에 등장하는 노드들의 PPR 합'으로 스코어링 -> top-k.
  (HippoRAG 의 '해마 인덱싱'을 우리 그래프에 이식: phrase 노드 대신 fused KG 의 타입형 노드,
   OpenIE passage 대신 우리 Chunk. PPR 로 멀티홉 연관을 single-step 회수한다.)

공정성:
  - 우리 방법과 **동일한 살아있는 B2 그래프**(retrieve_graphrag.GraphData)만 사용.
  - passage 뿐 아니라 PPR 상위 **엔티티 노드의 정형 속성**과 그들 사이 관계까지 컨텍스트에
    실어, 정형 절반을 눈감기지 않는다(HybridRAG/우리와 동일 정보 접근).
  - 합성 LLM·답변 규칙·채점기 모두 공유.
"""

from __future__ import annotations

from graph import EntityCatalog
from llm import LLMClient
from retrieve_graphrag import GraphData, EntityEmbeddingIndex

_NER_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
}
_NER_SYSTEM = (
    "Extract the named entities and specific noun phrases from the question that should be "
    "looked up in a manufacturing/supply-chain knowledge graph (ids like PRD001/SUP005/DT0007/"
    "PO.../EQ..., product/supplier/equipment names, and concrete concepts). "
    "Return JSON {\"entities\": [\"...\"]}. Do not include generic words."
)


class HippoRetriever:
    strategy_name = "HippoRAG"

    def __init__(self, gd: GraphData, llm: LLMClient, catalog: EntityCatalog,
                 emb_index: EntityEmbeddingIndex | None = None,
                 k_passages: int = 12, m_entities: int = 15,
                 alpha: float = 0.15, iters: int = 30, link_tau: float = 0.5,
                 link_margin: float = 0.05, debug: bool = False):
        self.gd = gd
        self.llm = llm
        self.catalog = catalog
        self.emb_index = emb_index
        self.k_passages = k_passages
        self.m_entities = m_entities
        self.alpha = alpha
        self.iters = iters
        # NER 구절의 임베딩-최근접 링킹 게이트.
        #  - link_tau   : 최소 코사인 floor(너무 약한 매칭 차단).
        #  - link_margin: top-1 이 top-2 보다 이만큼 더 가까울 때만 채택. generic 항('suppliers')
        #    은 수십 개 동종 노드에 거의 등거리라 margin≈0 -> 스킵(임의 허브 seed 방지).
        #    이름 변형처럼 한 노드만 뚜렷이 최근접이면 margin 이 커서 그대로 링킹(능력 보존).
        self.link_tau = link_tau
        self.link_margin = link_margin
        self.debug = debug

    # 1)+2) NER -> 노드 링킹
    def _link(self, question: str) -> list[str]:
        seeds = list(self.gd.link_seeds(question, self.catalog))     # 정확 ID/이름
        try:
            out = self.llm.chat_json(_NER_SYSTEM, f"Question: {question}", _NER_SCHEMA,
                                     list_key="entities", max_tokens=200)
            phrases = [p for p in (out.get("entities") or []) if isinstance(p, str) and p.strip()]
        except Exception as e:
            phrases = []
            if self.debug:
                print(f"  [hippo] NER 실패({type(e).__name__}) -> 정확매칭 seed 만 사용")
        # NER 환각 방어: 질문 본문에 실제 등장하는 구절만 채택(NER 은 질문에서 뽑는 것).
        qlow = question.lower()
        kept = [p for p in phrases if p.lower() in qlow]
        if self.debug:
            dropped = [p for p in phrases if p.lower() not in qlow]
            print(f"  [hippo] NER phrases={phrases}  질문에없어드롭={dropped}")
        for ph in kept:
            # (a) 정확 카탈로그 매칭(구절 안에 ID/이름이 그대로 있으면 잡힘)
            hit = self.gd.link_seeds(ph, self.catalog)
            if hit:
                seeds.extend(hit)
                continue
            # (b) 임베딩 최근접 엔티티: floor(코사인) + margin(모호성) 게이트.
            if self.emb_index is not None:
                near = self.emb_index.search_entities_scored(ph, k=2)
                if near:
                    eid, sim = near[0]
                    margin = sim - near[1][1] if len(near) > 1 else 1.0
                    take = sim >= self.link_tau and margin >= self.link_margin
                    if self.debug:
                        print(f"  [hippo] link {ph!r} -> {self.gd.nodes[eid]['id']} "
                              f"sim={sim:.3f} margin={margin:.3f} "
                              f"{'ADD' if take else 'SKIP(ambiguous)'}")
                    if take:
                        seeds.append(eid)
        return list(dict.fromkeys(seeds))

    def retrieve(self, question: str) -> dict:
        seeds = self._link(question)
        pr = self.gd.ppr(seeds, alpha=self.alpha, iters=self.iters)

        # 4) passage 스코어링: 그 청크에 등장하는 노드들의 PPR 합
        psg_scored = sorted(
            ((sum(pr.get(m, 0.0) for m in members), cid)
             for cid, members in self.gd.chunk_members.items()),
            reverse=True)
        passages = [cid for s, cid in psg_scored if s > 0][:self.k_passages]

        # 상위 엔티티(Document 제외) + 그들 사이 관계
        ent_scored = sorted(
            ((p, eid) for eid, p in pr.items()
             if p > 0 and self.gd.nodes[eid]["label"] != "Document"),
            reverse=True)
        top_ents = [eid for p, eid in ent_scored][:self.m_entities]
        top_set = set(top_ents)
        rels = [e for e in self.gd.edges if e["s"] in top_set and e["t"] in top_set]

        ent_ctx = "\n".join(f"  {self.gd.entity_text(e)}" for e in top_ents)
        rel_ctx = "\n".join(f"  {self.gd.edge_text(e)}" for e in rels)
        psg_ctx = "\n".join(f"  {self.gd.chunk_text(c)}" for c in passages)

        return {
            "ent_ctx": ent_ctx, "rel_ctx": rel_ctx, "psg_ctx": psg_ctx,
            "seeds": [self.gd.nodes[e]["id"] for e in seeds if e in self.gd.nodes],
            "passages": passages,
            "entities": [self.gd.nodes[e]["id"] for e in top_ents],
            "meta": {"n_seeds": len(seeds), "n_passages": len(passages),
                     "n_entities": len(top_ents), "n_rels": len(rels)},
        }
