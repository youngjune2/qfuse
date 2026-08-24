"""
chunk.py — 문서 본문을 1~3문장 슬라이딩 윈도우로 chunk화 (독립 모듈, 의존성 없음)
================================================================================
결정 반영:
  - 단위: 문장 기준, 윈도우 크기 최대 3문장, step 1 (겹침 허용).
  - 짧은 문서(<=3문장)는 chunk 1개.
  - 문서 전체 text 는 Document 노드에 그대로 보존(관계가 문장 경계를 넘는 경우 대비).

그래프 반영(파이프라인에서):
  (Document)-[:HAS_CHUNK]->(Chunk {chunk_id, seq, text, doc_id})
  Neo4j 내장 full-text 인덱스를 Chunk.text 에 건다(검색 때 원문 provenance 회수용).

chunk_id 규칙: "<doc_id>#c<seq>"  (예: DOC00258#c0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# 문장 분할: .?! 뒤 공백 기준. 약어까지 완벽히 처리하진 않지만(합성 영문 문서엔 충분),
# 필요하면 여기만 교체(예: spaCy/nltk)하면 된다.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    text: str
    def as_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    parts = _SENT_SPLIT.split(str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def sliding_windows(sentences: list[str], size: int = 3, step: int = 1) -> list[str]:
    """문장 리스트 -> 최대 size문장 윈도우들(step 간격). 문서가 size 이하면 통짜 1개."""
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(sentences)]
    out = []
    for start in range(0, len(sentences) - size + 1, step):
        out.append(" ".join(sentences[start:start + size]))
    # 마지막 윈도우가 끝 문장을 못 덮으면 보정
    last_start = len(sentences) - size
    if (last_start) % step != 0:
        out.append(" ".join(sentences[last_start:]))
    return out


def chunk_document(doc_id: str, text: str, size: int = 3, step: int = 1) -> list[Chunk]:
    windows = sliding_windows(split_sentences(text), size=size, step=step)
    return [Chunk(chunk_id=f"{doc_id}#c{i}", doc_id=doc_id, seq=i, text=w)
            for i, w in enumerate(windows)]


def chunk_corpus(documents: list[dict], text_col: str = "text", id_col: str = "document_id",
                 size: int = 3, step: int = 1) -> list[Chunk]:
    """documents: [{document_id, text, ...}, ...] -> 전체 Chunk 리스트."""
    chunks: list[Chunk] = []
    for d in documents:
        chunks.extend(chunk_document(str(d[id_col]), d.get(text_col, ""), size=size, step=step))
    return chunks


if __name__ == "__main__":
    demo = ("AE001 was recorded on EQ001 at 2026-07-05 06:30:00 as a high bearing vibration event. "
            "Root-cause analysis on EQ001 (Checkweigher 001) traced the failure to a defective "
            "precision bearing supplied by Harvest Consumer Goods 31 (SUP031). "
            "The fault triggered downtime DT001, which took the asset offline for 82 hours and "
            "halted production of Arts and Crafts Storage Box (PRD001) on line L003. "
            "SUP031 is a high-risk MRO supplier, and its 36-day lead time delayed delivery.")
    for c in chunk_document("DOC00258", demo):
        print(c.chunk_id, "|", c.text[:80], "...")
