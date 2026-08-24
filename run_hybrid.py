"""
run_hybrid.py — HybridRAG 베이스라인 실행/평가.

정형 KG(Text2Cypher, 정형 엣지만: retrieve_hybrid.VanillaStructuredRetriever) +
비정형 벡터(Chroma top-k 청크: vectorstore.VectorRetriever)를 각각 검색해, 두 결과를
하나의 합성 프롬프트에 모두 붙여(naive concat) 답을 만든다. Strategy D 와 동일한
eval 파일 / 채점기(run_query.grade) / 정확도 요약을 써서 직접 비교 가능하다.

전제:
  1) build_graph_neo4j.py 완료(정형 KG 가 Neo4j 에 있음)
  2) python vectorstore.py --build  완료(Chroma 인덱스 존재)

실행:
  python run_hybrid.py --eval eval_questions_bridge100.jsonl
  python run_hybrid.py --eval eval_questions_bridge100.jsonl --k 6 --outdir results_hybrid
  python run_hybrid.py --debug "For downtime DT0007, which supplier ...?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from graph import Neo4jClient
from llm import LLMClient
from retrieve_hybrid import VanillaStructuredRetriever
from vectorstore import VectorRetriever, format_chunks
from run_query import (grade, accuracy_summary, SYNTHESIS_RULES, SYNTHESIS_SYSTEM,
                       system_summary, print_system_summary)

# HybridRAG 합성: 두 출처(KG 정형 사실 + 문서 벡터 검색 구절)를 모두 쓰라고 명시.
# SYNTHESIS_RULES(근거/출처/폐쇄세계 부정 등)는 Strategy D 와 공유해 답변 규칙을 동일하게 유지.
# 격리 실험: system 지시문은 ours·graphrag 와 바이트 동일(SYNTHESIS_SYSTEM). concat 특유의
# 'TWO retrieval results (A)/(B)' 설명은 아래 user 메시지 레이아웃에만 둔다.
HYBRID_SYSTEM = SYNTHESIS_SYSTEM


def hybrid_synthesize(llm: LLMClient, question: str, kg_ctx: str, vec_ctx: str) -> str:
    user = (f"Question: {question}\n\n"
            f"[A. Knowledge graph results (structured tables, Text2Cypher)]\n{kg_ctx}\n\n"
            f"[B. Document passages (semantic search over documents)]\n{vec_ctx}")
    return llm.chat(HYBRID_SYSTEM, user, max_tokens=700)


def run_one(kg, vec, llm, question: str, item: dict | None = None) -> dict:
    t0 = time.perf_counter()
    try:
        r = kg.retrieve(question)
        hits = vec.retrieve(question)
        vec_ctx = format_chunks(hits)
        answer = hybrid_synthesize(llm, question, r.context_text, vec_ctx)
        return {
            "question": question,
            "type": (item or {}).get("type"),
            "case": (item or {}).get("case"),
            "gold_answer": (item or {}).get("answer"),
            "verdict": grade(answer, item),
            "expect": (item or {}).get("expect"),
            "seeds": r.meta.get("seeds"),
            "cypher": r.cypher,
            "kg_context": r.context_text,
            "vector_hits": hits,
            "vector_context": vec_ctx,
            "answer": answer,
            "attempts": r.meta.get("attempts"),
            "kg_error": r.meta.get("error"),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        return {
            "question": question,
            "type": (item or {}).get("type"),
            "case": (item or {}).get("case"),
            "gold_answer": (item or {}).get("answer"),
            "verdict": "ERROR",
            "expect": (item or {}).get("expect"),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }


def show(res: dict, debug: bool):
    print(f"\n[Q] {res['question']}")
    if res.get("verdict"):
        print(f"    [{res['verdict']}] {res.get('case') or ''}  (gold={res.get('gold_answer')})")
    if res.get("error"):
        print(f"    ERROR: {res['error']}")
    if debug:
        print(f"    seeds: {res.get('seeds')}")
        print("\n--- KG cypher ---")
        print(res.get("cypher") or "(none)")
        print("\n--- KG context ---")
        print((res.get("kg_context") or "")[:1200])
        print("\n--- vector context ---")
        print((res.get("vector_context") or "")[:1500])
    print("\n--- answer ---")
    print(res.get("answer") or "(none)")
    print()


def build_markdown(results: list[dict], header: dict) -> str:
    out = ["# HybridRAG 베이스라인 결과 (정형 KG + 벡터 concat)\n"]
    out.append(f"- 생성 시각: {header['timestamp']}")
    out.append(f"- 문항 수: {len(results)}")
    out.append(f"- 벡터 top-k: {header.get('k')}  ·  KG 정형 관계타입: {header.get('n_structured_types')}")
    acc = header.get("accuracy_summary")
    if acc and acc.get("graded"):
        out.append(f"- **정확도: {acc['pass']}/{acc['graded']} = {acc['accuracy']}**  (ERROR {acc['errors']})")
        for t, (p, tot) in sorted(acc["by_type"].items()):
            out.append(f"    - `{t}`: {p}/{tot}")
    sysm = header.get("system_summary")
    if sysm and sysm.get("n"):
        lat, ca = sysm.get("latency_sec", {}), sysm.get("cypher_attempts", {})
        out.append(f"- **시스템**: latency mean {lat.get('mean')}s / p95 {lat.get('p95')}s"
                   f"  ·  KG cypher attempts mean {ca.get('mean')} (retry {ca.get('retry_rate')})"
                   f"  ·  error_rate {sysm.get('error_rate')}")
    out.append("")
    for i, r in enumerate(results, 1):
        out.append("\n---\n")
        out.append(f"## Q{i}. {r['question']}\n")
        if r.get("verdict"):
            out.append(f"- verdict: **{r['verdict']}**  (case `{r.get('case')}`, gold `{r.get('gold_answer')}`)")
        if r.get("type"):
            out.append(f"- type: `{r['type']}`")
        out.append(f"- elapsed: {r.get('elapsed_sec')}s")
        if r.get("error"):
            out.append(f"\n> ⚠️ ERROR: `{r['error']}`")
        out.append(f"- seeds: `{r.get('seeds')}`  ·  KG attempts: {r.get('attempts')}")
        out.append("\n**KG Cypher (정형)**")
        out.append("```cypher\n" + (r.get("cypher") or "(none)") + "\n```")
        out.append("\n**KG context**")
        out.append("```\n" + (r.get("kg_context") or "(none)") + "\n```")
        out.append("\n**Vector 검색(top-k 청크)**")
        out.append("```\n" + (r.get("vector_context") or "(none)") + "\n```")
        out.append("\n**Answer**\n")
        out.append(r.get("answer") or "(none)")
    return "\n".join(out) + "\n"


def save(results: list[dict], header: dict, outdir: str) -> tuple[Path, Path]:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    ts = header["timestamp"]
    jp = d / f"hybrid_{ts}.json"
    mp = d / f"hybrid_{ts}.md"
    jp.write_text(json.dumps({"header": header, "results": results},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp.write_text(build_markdown(results, header), encoding="utf-8")
    return jp, mp


def main() -> int:
    ap = argparse.ArgumentParser(description="HybridRAG 베이스라인(정형 KG + 벡터)")
    ap.add_argument("question", nargs="*", help="단건 질문")
    ap.add_argument("--eval", help="jsonl 배치 실행 + 자동 채점")
    ap.add_argument("--k", type=int, default=6, help="벡터 top-k (기본 6)")
    ap.add_argument("--tau", type=float, default=0.5, help="(KG 실행기 공유 파라미터)")
    ap.add_argument("--retries", type=int, default=2, help="Cypher 재생성 횟수")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--outdir", default="results_hybrid", help="결과 저장 폴더")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    graph = Neo4jClient()
    llm = LLMClient()
    catalog = graph.load_entity_catalog()
    kg = VanillaStructuredRetriever(graph, llm, catalog, tau=args.tau,
                                    retries=args.retries, debug=args.debug)
    vec = VectorRetriever(k=args.k)
    print(f"카탈로그 {len(catalog.by_id)} 엔티티 / 정형 관계타입 {len(kg.all_types)}종 / 벡터 top-k {args.k}")

    header = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": "HybridRAG (structured-KG Text2Cypher + Chroma vector, prompt concat)",
        "catalog_size": len(catalog.by_id),
        "n_structured_types": len(kg.all_types),
        "structured_types": kg.all_types,
        "k": args.k,
        "tau": args.tau,
    }
    results: list[dict] = []
    try:
        if args.eval:
            items = [json.loads(l) for l in
                     Path(args.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, it in enumerate(items, 1):
                print("=" * 70)
                print(f"Q{i}/{len(items)} [{it.get('type','')}]")
                res = run_one(kg, vec, llm, it.get("q", ""), it)
                show(res, args.debug)
                results.append(res)
        else:
            if not args.question:
                ap.error("질문을 입력하거나 --eval 을 쓰세요.")
            res = run_one(kg, vec, llm, " ".join(args.question))
            show(res, args.debug)
            results.append(res)
    finally:
        graph.close()

    acc = accuracy_summary(results)
    header["accuracy_summary"] = acc
    if acc.get("graded"):
        print("=" * 70)
        print(f"[정확도] {acc['pass']}/{acc['graded']} = {acc['accuracy']}  (ERROR {acc['errors']})")
        for t, (p, tot) in sorted(acc["by_type"].items()):
            print(f"    {t:<16} {p}/{tot}")

    header["system_summary"] = system_summary(results)
    print_system_summary(header["system_summary"])

    if results and not args.no_save:
        jp, mp = save(results, header, args.outdir)
        print("=" * 70)
        print(f"저장 완료:\n  {jp}\n  {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
