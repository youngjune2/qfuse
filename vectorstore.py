"""
vectorstore.py — HybridRAG의 '벡터 반쪽'.

비정형 문서(documents.csv)를 문장 슬라이딩 윈도우로 청크(chunk.py) -> vLLM 임베딩
엔드포인트(:8001, llm.EmbeddingClient)로 임베딩 -> Chroma(PersistentClient)에 적재/검색.

정형 KG(Text2Cypher)와 대비되는 '순수 벡터 RAG' 컴포넌트다. run_hybrid.py 가 이 검색
결과와 KG 검색 결과를 한 프롬프트에 함께 붙여 HybridRAG 베이스라인을 만든다.

의존성:  pip install chromadb   (pandas/openai 는 기존 파이프라인과 동일)

빌드(1회, 문서/청크 파라미터 바뀔 때만 다시):
  python vectorstore.py --build
검색 확인:
  python vectorstore.py --query "For downtime DT0007, which supplier ..." --k 6
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from chunk import chunk_corpus
from llm import EmbeddingClient

# build_graph_neo4j.py 와 동일한 env 규칙을 따른다(같은 문서 소스를 청크한다).
DATA_DIR       = os.getenv("DATA_DIR", "data")
DOCUMENTS_FILE = os.getenv("DOCUMENTS_FILE", "documents.csv")

CHROMA_DIR     = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION     = os.getenv("CHROMA_COLLECTION", "doc_chunks")
EMB_BATCH      = int(os.getenv("EMB_BATCH", "16"))


def _load_documents() -> list[dict]:
    df = pd.read_csv(os.path.join(DATA_DIR, DOCUMENTS_FILE), encoding="utf-8-sig")
    return df.to_dict("records")


def _embed_batch(emb: EmbeddingClient, texts: list[str]) -> list[list[float]]:
    """한 배치 임베딩. 서버가 배치 크기/토큰량으로 5xx 를 내면 반씩 쪼개 재시도한다
    (서버가 감당하는 크기에 자동 적응). 단일 항목까지 줄여도 실패하면 그 텍스트를 알리고 예외."""
    try:
        return emb.embed(texts)
    except Exception as e:
        if len(texts) <= 1:
            print(f"  [error] 단일 텍스트 임베딩 실패({type(e).__name__}): {texts[0][:120]!r}")
            raise
        mid = len(texts) // 2
        return _embed_batch(emb, texts[:mid]) + _embed_batch(emb, texts[mid:])


def _embed_all(emb: EmbeddingClient, texts: list[str]) -> list[list[float]]:
    """엔드포인트 부하를 고려해 배치로 임베딩(순서 보존)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMB_BATCH):
        out.extend(_embed_batch(emb, texts[i:i + EMB_BATCH]))
        print(f"  임베딩 {min(i + EMB_BATCH, len(texts))}/{len(texts)}")
    return out


def build_index(size: int = 3, step: int = 1):
    """문서 -> 청크 -> 임베딩 -> Chroma 컬렉션(재빌드 시 기존 컬렉션 제거)."""
    import chromadb

    docs = _load_documents()
    chunks = chunk_corpus(docs, text_col="text", id_col="document_id", size=size, step=step)
    texts = [c.text for c in chunks]
    print(f"문서 {len(docs)}개 -> 청크 {len(chunks)}개 (size={size}, step={step}). 임베딩 시작...")

    emb = EmbeddingClient()
    vecs = _embed_all(emb, texts)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)   # 재빌드: 잔여 데이터 제거
    except Exception:
        pass
    # 임베딩을 명시적으로 넣으므로 Chroma 내장 임베딩함수(기본 ONNX MiniLM 다운로드)는 끈다.
    # 코사인 공간 지정. 질의도 query_embeddings 로 직접 넣으므로 EF 없이 동작한다.
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                   embedding_function=None)
    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vecs,
        documents=texts,
        metadatas=[{"doc_id": c.doc_id, "seq": c.seq} for c in chunks],
    )
    print(f"Chroma 적재 완료: path={CHROMA_DIR}  collection={COLLECTION}  chunks={len(chunks)}")
    return col


class VectorRetriever:
    """Chroma 컬렉션에서 질문 top-k 청크를 회수. 질문도 동일 vLLM 엔드포인트로 임베딩한다."""

    def __init__(self, k: int = 6):
        import chromadb

        self.k = k
        self.emb = EmbeddingClient()
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_collection(COLLECTION, embedding_function=None)

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        qv = self.emb.embed([question])[0]
        res = self.col.query(query_embeddings=[qv], n_results=k,
                             include=["documents", "metadatas", "distances"])
        ids   = res["ids"][0]
        docs  = res["documents"][0]
        dists = (res.get("distances") or [[None] * len(ids)])[0]
        return [{"chunk_id": i, "text": t, "distance": d}
                for i, t, d in zip(ids, docs, dists)]


def format_chunks(hits: list[dict]) -> str:
    """합성 프롬프트에 붙일 top-k 청크 블록."""
    if not hits:
        return "(no passages retrieved)"
    return "\n".join(f"  [{h['chunk_id']}] {h['text']}" for h in hits)


def main() -> int:
    ap = argparse.ArgumentParser(description="HybridRAG 벡터 인덱스(Chroma) 빌드/검색")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="문서 청크 임베딩 -> Chroma 적재")
    g.add_argument("--query", help="검색 테스트 질의")
    ap.add_argument("--k", type=int, default=6, help="top-k (검색 테스트)")
    ap.add_argument("--size", type=int, default=3, help="청크 윈도우 문장 수(빌드)")
    ap.add_argument("--step", type=int, default=1, help="청크 슬라이딩 step(빌드)")
    args = ap.parse_args()

    if args.build:
        build_index(size=args.size, step=args.step)
    else:
        vr = VectorRetriever(k=args.k)
        for h in vr.retrieve(args.query):
            d = h["distance"]
            print(f"[{h['chunk_id']}] dist={d:.4f}" if d is not None else f"[{h['chunk_id']}]")
            print(f"    {h['text'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
