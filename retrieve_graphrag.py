"""
retrieve_graphrag.py — HippoRAG / LightRAG 베이스라인의 공유 토대.

대조 프레임(사용자 재정의): fused KG 자체는 이미 존재하므로 'unify'는 독창성이 아니다.
따라서 강점은 **검색 방법**에서 나와야 한다. 이 파일은 우리 방법(retrieve_d)이 쓰는 것과
**완전히 동일한 살아있는 B2 그래프**를 메모리로 올려, 그 위에서 HippoRAG(PPR)·LightRAG
(dual-level 키워드)를 돌리게 한다. 세 방법의 유일한 차이가 '검색 알고리즘'이 되도록 격리한다.

공정성 원칙(strawman 방지):
  1) 같은 그래프    : Neo4j 의 동일 B2 fused 그래프(정형 backbone + document_link + extracted).
  2) 같은 LLM       : llm.LLMClient (NER/키워드 추출·최종 합성 모두). 같은 SYNTHESIS_RULES.
  3) 같은 채점/지표 : run_query.grade / system_summary / 같은 eval 파일.
  4) 넉넉한 예산    : recall 기반 방법이 굶지 않도록 top-k 를 크게, configurable.
  5) 정형 미차단    : passages=문서 Chunk 지만, 검색이 도달한 **엔티티 노드의 속성(정형 레코드)**
                     을 컨텍스트에 함께 실어 정형 절반을 눈감기지 않는다.
  6) 충실한 메커니즘: PPR 은 외부 의존성 없이 자체 power-iteration 으로 구현(환경 리스크 제거).

이 파일이 제공하는 것:
  - GraphData      : 노드/엣지/청크/멤버십/인접리스트 + PPR + 시드링킹 + 직렬화.
  - EntityEmbeddingIndex : 엔티티/관계타입 임베딩 인덱스(LightRAG 키워드 매칭, 디스크 캐시).
  - graphrag_synthesize  : 엔티티+관계+구절을 한 프롬프트로 합성(HybridRAG 와 동형).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections import defaultdict

from graph import Neo4jClient, EntityCatalog, LABEL_PK
from llm import LLMClient, EmbeddingClient, EMB_MODEL
from schema import KEY_PROPS, build_rel_glosses, relationship_catalog
from run_query import SYNTHESIS_RULES, SYNTHESIS_SYSTEM

_NAME_COLS = ("product_name", "supplier_name", "equipment_name", "line_name",
              "region_name", "customer_name", "title")
_ID_TOKEN = re.compile(r"[A-Za-z]{2,6}\d{2,}")
_CACHE_DIR = os.getenv("GRAPHRAG_CACHE", "./graphrag_cache")


# ---------------------------------------------------------------------------
# 그래프 적재 (retrieve_d 와 동일한 살아있는 그래프)
# ---------------------------------------------------------------------------
class GraphData:
    """살아있는 B2 그래프를 메모리로. 노드/엣지/청크/멤버십/무방향 인접리스트."""

    def __init__(self, graph: Neo4jClient, tau: float = 0.5, debug: bool = False):
        self.graph = graph
        self.tau = tau
        self.debug = debug
        self._load()

    def _q(self, cypher: str):
        return self.graph.run_read(cypher, validate=False)

    def _load(self):
        # 1) 노드(청크 제외). elementId 를 PPR 노드키로 쓴다(세션 내 안정).
        self.nodes: dict[str, dict] = {}
        self.idkey_to_eid: dict[tuple[str, str], str] = {}
        self.id_to_eid: dict[str, str] = {}
        for r in self._q("MATCH (n) WHERE NOT n:Chunk "
                         "RETURN elementId(n) AS eid, labels(n)[0] AS label, properties(n) AS props"):
            eid, label, props = r["eid"], r["label"], (r["props"] or {})
            pk = LABEL_PK.get(label)
            nid = str(props.get(pk)) if pk and props.get(pk) is not None else eid
            name = next((str(props[c]) for c in _NAME_COLS if props.get(c)), None)
            self.nodes[eid] = {"label": label, "id": nid, "name": name, "props": props}
            if pk:
                self.idkey_to_eid[(label, nid)] = eid
            self.id_to_eid.setdefault(nid, eid)

        # 2) 엣지(청크 제외). 무방향 인접리스트(PPR)와 원본 엣지 목록.
        self.edges: list[dict] = []
        self.adj: dict[str, list[str]] = defaultdict(list)
        self.incident: dict[str, list[int]] = defaultdict(list)   # eid -> edge index 목록
        for r in self._q(
            "MATCH (a)-[r]->(b) WHERE NOT a:Chunk AND NOT b:Chunk "
            "RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS type, "
            "coalesce(r.origin,'?') AS origin, r.confidence AS cf, "
            "r.source_chunk_id AS sc, r.evidence AS ev"
        ):
            s, t = r["s"], r["t"]
            if s not in self.nodes or t not in self.nodes:
                continue
            idx = len(self.edges)
            self.edges.append({"s": s, "t": t, "type": r["type"], "origin": r["origin"],
                               "cf": r["cf"], "sc": r["sc"], "ev": r["ev"]})
            self.adj[s].append(t)
            self.adj[t].append(s)
            self.incident[s].append(idx)
            self.incident[t].append(idx)

        # 3) 청크(문서 passages).
        self.chunks: dict[str, dict] = {
            r["id"]: {"doc": r["doc"], "text": r["text"] or ""}
            for r in self._q("MATCH (c:Chunk) RETURN c.chunk_id AS id, c.doc_id AS doc, c.text AS text")
            if r["id"]
        }

        # 4) 청크 멤버십(chunk_id -> 그 청크에 등장/연결된 노드 eid 집합).
        #    (a) extracted 엣지의 source_chunk_id: 그 청크가 양끝 엔티티를 언급.
        #    (b) document_link: Document -> entity. 그 Document 의 모든 청크가 그 엔티티를 담음.
        #    (c) id-토큰 부분매칭: 청크 본문에 등장하는 엔티티 ID(HippoRAG phrase-in-passage 와 동형).
        docid_to_eid = {n["id"]: eid for eid, n in self.nodes.items() if n["label"] == "Document"}
        doc_targets: dict[str, set] = defaultdict(set)
        for e in self.edges:
            if e["origin"] == "document_link":
                doc_targets[e["s"]].add(e["t"])
        self.chunk_members: dict[str, set] = defaultdict(set)
        for e in self.edges:
            if e["origin"] == "extracted" and e["sc"] and e["sc"] in self.chunks:
                self.chunk_members[e["sc"]].add(e["s"])
                self.chunk_members[e["sc"]].add(e["t"])
        for cid, c in self.chunks.items():
            deid = docid_to_eid.get(str(c["doc"]))
            if deid is not None:
                self.chunk_members[cid].add(deid)
                self.chunk_members[cid].update(doc_targets.get(deid, ()))
            for tok in set(_ID_TOKEN.findall(c["text"])):
                eid = self.id_to_eid.get(tok)
                if eid is not None:
                    self.chunk_members[cid].add(eid)

        # 5) 역인덱스(노드 -> 그 노드가 등장하는 청크).
        self.node_chunks: dict[str, set] = defaultdict(set)
        for cid, members in self.chunk_members.items():
            for eid in members:
                self.node_chunks[eid].add(cid)

        self.rel_glosses = build_rel_glosses(self.graph, relationship_catalog(self.graph))
        if self.debug:
            print(f"  [GraphData] 노드 {len(self.nodes)} / 엣지 {len(self.edges)} / "
                  f"청크 {len(self.chunks)} / 관계타입 {len({e['type'] for e in self.edges})}")

    # --- 질문 시드 링킹(정확 매칭: ID + 이름). 세 방법 공통 앵커 ---
    def link_seeds(self, question: str, catalog: EntityCatalog) -> list[str]:
        out: list[str] = []
        for (label, id_) in catalog.find_in_text(question):
            eid = self.idkey_to_eid.get((label, str(id_)))
            if eid:
                out.append(eid)
        return list(dict.fromkeys(out))

    # --- Personalized PageRank (외부 의존성 없는 power-iteration) ---
    def ppr(self, seed_eids: list[str], alpha: float = 0.15, iters: int = 30) -> dict[str, float]:
        seeds = [e for e in dict.fromkeys(seed_eids) if e in self.nodes]
        if not seeds:
            return {}
        teleport = {e: 1.0 / len(seeds) for e in seeds}
        nodes = list(self.nodes.keys())
        deg = {e: len(self.adj[e]) for e in nodes}
        pr = {e: teleport.get(e, 0.0) for e in nodes}
        for _ in range(iters):
            nxt = {e: alpha * teleport.get(e, 0.0) for e in nodes}
            dangling = 0.0
            for e in nodes:
                p = pr[e]
                if p == 0.0:
                    continue
                d = deg[e]
                if d == 0:
                    dangling += p
                    continue
                share = (1.0 - alpha) * p / d
                for nb in self.adj[e]:
                    nxt[nb] += share
            if dangling:                          # dangling 질량은 teleport 분포로 환류
                for s, w in teleport.items():
                    nxt[s] += (1.0 - alpha) * dangling * w
            pr = nxt
        return pr

    # --- 직렬화 헬퍼 ---
    def entity_text(self, eid: str) -> str:
        n = self.nodes[eid]
        keyprops = KEY_PROPS.get(n["label"], "")
        keys = [k.strip() for k in keyprops.split(",") if k.strip()] or list(n["props"].keys())
        fields = []
        for k in keys:
            if k == "text":                        # Document.text 전체 덤프 방지
                continue
            v = n["props"].get(k)
            if v is not None and v != "":
                fields.append(f"{k}={v}")
        return f"({n['label']}) " + ", ".join(fields)

    def edge_text(self, e: dict) -> str:
        a, b = self.nodes[e["s"]], self.nodes[e["t"]]
        line = f"({a['label']} {a['id']})-[:{e['type']}]->({b['label']} {b['id']})  origin={e['origin']}"
        if e["origin"] == "extracted":
            if e["cf"] is not None:
                line += f" conf={e['cf']}"
            if e["ev"]:
                line += f"  // {str(e['ev'])[:100]}"
        else:
            g = self.rel_glosses.get(e["type"])
            if g:
                line += f"  // {g}"
        return line

    def chunk_text(self, cid: str) -> str:
        return f"[{cid}] {self.chunks.get(cid, {}).get('text', '')}"


# ---------------------------------------------------------------------------
# 임베딩 인덱스 (LightRAG 키워드 매칭 · 선택적 HippoRAG 이름링킹)
# ---------------------------------------------------------------------------
class EntityEmbeddingIndex:
    """엔티티(정형 비즈니스 노드)와 관계타입을 임베딩해 코사인 top-k 검색.
    Document/Chunk 는 엔티티 인덱스에서 제외(LightRAG low-level=엔티티 매칭 취지).
    벡터는 콘텐츠 해시로 디스크 캐시(두 방법·재실행 간 재사용)."""

    def __init__(self, gd: GraphData, batch: int = 16, debug: bool = False):
        import numpy as np
        self.np = np
        self.gd = gd
        self.emb = EmbeddingClient()
        self.debug = debug
        # 엔티티 후보(Document 제외)
        self.ent_eids = [eid for eid, n in gd.nodes.items() if n["label"] != "Document"]
        ent_texts = [self._ent_str(eid) for eid in self.ent_eids]
        # 관계타입 후보(gloss 있으면 gloss, 없으면 humanized)
        self.types = sorted({e["type"] for e in gd.edges})
        type_texts = [self._type_str(t) for t in self.types]
        self.ent_mat = self._embed_cached("entities", ent_texts, batch)
        self.type_mat = self._embed_cached("types", type_texts, batch)

    def _ent_str(self, eid: str) -> str:
        n = self.gd.nodes[eid]
        return f"{n['label']} {n['id']}" + (f" {n['name']}" if n["name"] else "")

    def _type_str(self, t: str) -> str:
        g = self.gd.rel_glosses.get(t)
        return f"{t.replace('_', ' ').lower()}" + (f": {g}" if g else "")

    def _embed_cached(self, tag: str, texts: list[str], batch: int):
        np = self.np
        os.makedirs(_CACHE_DIR, exist_ok=True)
        key = hashlib.md5(("|".join(texts) + "||" + EMB_MODEL).encode("utf-8")).hexdigest()
        path = os.path.join(_CACHE_DIR, f"{tag}_{key}.pkl")
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                mat = pickle.load(fh)
            if self.debug:
                print(f"  [emb-cache] {tag}: {mat.shape} (cached)")
            return mat
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch):
            vecs.extend(self._embed_batch(texts[i:i + batch]))
        mat = np.asarray(vecs, dtype="float32") if vecs else np.zeros((0, 0), dtype="float32")
        if mat.size:
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        with open(path, "wb") as fh:
            pickle.dump(mat, fh)
        if self.debug:
            print(f"  [emb] {tag}: {mat.shape} (computed & cached)")
        return mat

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """서버가 배치로 5xx 를 내면 반씩 쪼개 재시도(vectorstore 와 동일 전략)."""
        try:
            return self.emb.embed(texts)
        except Exception:
            if len(texts) <= 1:
                raise
            mid = len(texts) // 2
            return self._embed_batch(texts[:mid]) + self._embed_batch(texts[mid:])

    def _search_scored(self, mat, ids: list, query: str, k: int) -> list[tuple]:
        np = self.np
        if mat is None or mat.size == 0 or not ids:
            return []
        qv = np.asarray(self.emb.embed([query])[0], dtype="float32")
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = mat @ qv                            # mat 은 정규화돼 있으므로 코사인
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]

    def _search(self, mat, ids: list, query: str, k: int) -> list:
        return [eid for eid, _ in self._search_scored(mat, ids, query, k)]

    def search_entities(self, keyword: str, k: int = 5) -> list[str]:
        return self._search(self.ent_mat, self.ent_eids, keyword, k)

    def search_entities_gated(self, keyword: str, k: int = 5, tau: float = 0.5,
                              margin: float = 0.05) -> list[str]:
        """floor(tau) + 모호성(margin) 게이트 통과 시에만 엔티티 반환. generic 키워드
        ('suppliers')는 동종 노드에 거의 등거리라 margin≈0 -> 빈 리스트(노이즈 차단).
        명확히 한 엔티티로 해소되면 그 상위 매칭들을 반환. HippoRAG 링킹과 동일 원칙."""
        scored = self._search_scored(self.ent_mat, self.ent_eids, keyword, max(k + 1, 2))
        if not scored or scored[0][1] < tau:
            return []
        if len(scored) > 1 and (scored[0][1] - scored[1][1]) < margin:
            return []
        return [eid for eid, s in scored[:k] if s >= tau]

    def search_entities_scored(self, keyword: str, k: int = 1) -> list[tuple]:
        """(eid, cosine) top-k. HippoRAG NER 링킹의 임계값 게이팅용."""
        return self._search_scored(self.ent_mat, self.ent_eids, keyword, k)

    def search_types(self, keyword: str, k: int = 3) -> list[str]:
        return self._search(self.type_mat, self.types, keyword, k)


# ---------------------------------------------------------------------------
# 합성 (HybridRAG 와 동형: 근거/출처/폐쇄세계 규칙 공유)
# ---------------------------------------------------------------------------
# 격리 실험: system 지시문은 ours·hybrid 와 바이트 동일(SYNTHESIS_SYSTEM). 검색기별 차이는
# 아래 user 메시지의 A/B/C context 레이아웃에만 존재. (구 preamble의 'Use all three' 힌트는
# graphrag 에만 주어져 베이스라인에 유리했으므로 제거 — 격리 위해 정본으로 통일.)
GRAPHRAG_SYSTEM = SYNTHESIS_SYSTEM


def graphrag_synthesize(llm: LLMClient, question: str, ent_ctx: str, rel_ctx: str,
                        psg_ctx: str) -> str:
    user = (f"Question: {question}\n\n"
            f"[A. Entities (structured fields)]\n{ent_ctx or '(none)'}\n\n"
            f"[B. Relationships]\n{rel_ctx or '(none)'}\n\n"
            f"[C. Document passages]\n{psg_ctx or '(none)'}")
    return llm.chat(GRAPHRAG_SYSTEM, user, max_tokens=700)
