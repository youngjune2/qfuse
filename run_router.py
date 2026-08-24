"""
run_router.py — baseline #1(rule-driven router) 실행/평가.

route.RuleRouter 로 질문을 4-way(DB/Doc/Hybrid/LLM) 라우팅한 뒤, 선택된 경로만 실행해서
답을 만든다. 경로별 실행기는 기존 컴포넌트를 그대로 재사용해 비교를 공정하게 유지한다:
  - DB     : retrieve_hybrid.VanillaStructuredRetriever (정형 origin 만, 통짜 Text2Cypher)
  - Doc    : vectorstore.VectorRetriever (Chroma top-k 청크)
  - Hybrid : 위 둘 다 + run_hybrid.hybrid_synthesize (naive concat)
  - LLM    : 검색 없이 모델이 직접 답변
채점기/정확도/시스템 지표는 Strategy D·HybridRAG 와 동일(run_query) -> 직접 비교 가능.

대조 논지: "통합(Unify) 대신 소스별 라우팅(Route)" 이 bridge 질문에서 어떻게 무너지는지를
라우팅 분포 + 경로별 정확도로 드러낸다.

전제:
  1) build_graph_neo4j.py 완료(정형 KG in Neo4j)      -> DB/Hybrid 경로
  2) python vectorstore.py --build 완료(Chroma 인덱스) -> Doc/Hybrid 경로

실행:
  python run_router.py --eval eval_questions_bridge100.jsonl
  python run_router.py --debug "For downtime DT0007, which supplier ...?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from graph import Neo4jClient
from llm import LLMClient
from retrieve_hybrid import VanillaStructuredRetriever
from vectorstore import VectorRetriever, format_chunks
from route import RuleRouter, PATHS
from run_query import (grade, accuracy_summary, system_summary, print_system_summary,
                       synthesize, SYNTHESIS_RULES)
from run_hybrid import hybrid_synthesize

_DOC_SYSTEM = ("You are a manufacturing/supply-chain analyst answering ONLY from retrieved company "
               "document passages (semantic search).\n\n" + SYNTHESIS_RULES)
_LLM_SYSTEM = ("You are a manufacturing/supply-chain analyst. Answer from your own general knowledge; "
               "no company data was retrieved. If the question needs a specific company record you "
               "were not given, say you cannot find it. Answer concisely in the question's language.")


def _doc_synthesize(llm: LLMClient, question: str, vec_ctx: str) -> str:
    return llm.chat(_DOC_SYSTEM, f"Question: {question}\n\n[Document passages]\n{vec_ctx}", max_tokens=700)


def _direct_answer(llm: LLMClient, question: str) -> str:
    return llm.chat(_LLM_SYSTEM, f"Question: {question}", max_tokens=400)


def run_one(router, kg, vec, llm, question: str, item: dict | None = None) -> dict:
    t0 = time.perf_counter()
    base = {
        "question": question,
        "type": (item or {}).get("type"),
        "case": (item or {}).get("case"),
        "gold_answer": (item or {}).get("answer"),
        "expect": (item or {}).get("expect"),
    }
    try:
        dec = router.route(question)
        path = dec["path"]
        cypher = kg_ctx = vec_ctx = None
        hits = None
        attempts = None
        kg_error = None

        if path == "DB":
            r = kg.retrieve(question)
            cypher, kg_ctx = r.cypher, r.context_text
            attempts, kg_error = r.meta.get("attempts"), r.meta.get("error")
            answer = synthesize(llm, question, r)
        elif path == "Doc":
            hits = vec.retrieve(question)
            vec_ctx = format_chunks(hits)
            answer = _doc_synthesize(llm, question, vec_ctx)
        elif path == "Hybrid":
            r = kg.retrieve(question)
            cypher, kg_ctx = r.cypher, r.context_text
            attempts, kg_error = r.meta.get("attempts"), r.meta.get("error")
            hits = vec.retrieve(question)
            vec_ctx = format_chunks(hits)
            answer = hybrid_synthesize(llm, question, kg_ctx, vec_ctx)
        else:  # LLM
            answer = _direct_answer(llm, question)

        return {**base,
                "route": path,
                "route_scores": dec["scores"],
                "route_reasoning": dec["reasoning"],
                "verdict": grade(answer, item),
                "cypher": cypher,
                "kg_context": kg_ctx,
                "vector_hits": hits,
                "vector_context": vec_ctx,
                "answer": answer,
                "attempts": attempts,
                "kg_error": kg_error,
                "elapsed_sec": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        return {**base,
                "route": locals().get("path"),
                "verdict": "ERROR",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "elapsed_sec": round(time.perf_counter() - t0, 2)}


def routing_summary(results: list[dict]) -> dict:
    """라우팅 분포 + 경로별 정확도 + gold-type × 경로 교차표(대조 분석의 핵심)."""
    dist = defaultdict(int)
    by_path = defaultdict(lambda: [0, 0])          # path -> [pass, graded]
    type_path = defaultdict(lambda: defaultdict(int))   # type -> {path: n}
    for r in results:
        p = r.get("route") or "?"
        dist[p] += 1
        type_path[r.get("type") or "?"][p] += 1
        if r.get("verdict") in ("PASS", "FAIL"):
            b = by_path[p]
            b[1] += 1
            if r["verdict"] == "PASS":
                b[0] += 1
    return {
        "distribution": dict(dist),
        "accuracy_by_path": {p: {"pass": v[0], "graded": v[1],
                                 "accuracy": round(v[0] / v[1], 4) if v[1] else None}
                             for p, v in by_path.items()},
        "type_x_path": {t: dict(d) for t, d in type_path.items()},
    }


def print_routing_summary(rs: dict):
    print("=" * 70)
    print("[라우팅 분포]  " + "  ".join(f"{p}={rs['distribution'].get(p, 0)}" for p in PATHS))
    print("[경로별 정확도]")
    for p in PATHS:
        a = rs["accuracy_by_path"].get(p)
        if a and a["graded"]:
            print(f"    {p:<7} {a['pass']}/{a['graded']} = {a['accuracy']}")


def show(res: dict, debug: bool):
    print(f"\n[Q] {res['question']}")
    print(f"    -> route: {res.get('route')}  scores={res.get('route_scores')}")
    if res.get("verdict"):
        print(f"    [{res['verdict']}] {res.get('case') or ''}  (gold={res.get('gold_answer')})")
    if res.get("error"):
        print(f"    ERROR: {res['error']}")
    if debug:
        print(f"    reasoning: {res.get('route_reasoning')}")
        if res.get("cypher"):
            print("\n--- KG cypher ---\n" + res["cypher"])
        if res.get("vector_context"):
            print("\n--- vector context ---\n" + res["vector_context"][:1200])
    print("\n--- answer ---")
    print(res.get("answer") or "(none)")
    print()


def build_markdown(results: list[dict], header: dict) -> str:
    out = ["# Rule-Driven Router 베이스라인 결과 (Route vs Unify 대조군)\n"]
    out.append(f"- 생성 시각: {header['timestamp']}")
    out.append(f"- 문항 수: {len(results)}  ·  벡터 top-k: {header.get('k')}")
    acc = header.get("accuracy_summary")
    if acc and acc.get("graded"):
        out.append(f"- **정확도: {acc['pass']}/{acc['graded']} = {acc['accuracy']}**  (ERROR {acc['errors']})")
        for t, (p, tot) in sorted(acc["by_type"].items()):
            out.append(f"    - `{t}`: {p}/{tot}")
    rs = header.get("routing_summary")
    if rs:
        out.append("- **라우팅 분포**: " + ", ".join(f"{p}={rs['distribution'].get(p, 0)}" for p in PATHS))
        out.append("- **경로별 정확도**:")
        for p in PATHS:
            a = rs["accuracy_by_path"].get(p)
            if a and a["graded"]:
                out.append(f"    - `{p}`: {a['pass']}/{a['graded']} = {a['accuracy']}")
        out.append("- **gold-type × route 교차표**:")
        for t, d in sorted(rs["type_x_path"].items()):
            out.append(f"    - `{t}`: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    sysm = header.get("system_summary")
    if sysm and sysm.get("n"):
        lat = sysm.get("latency_sec", {})
        out.append(f"- **시스템**: latency mean {lat.get('mean')}s / p95 {lat.get('p95')}s"
                   f"  ·  error_rate {sysm.get('error_rate')}")
    out.append("")
    for i, r in enumerate(results, 1):
        out.append("\n---\n")
        out.append(f"## Q{i}. {r['question']}\n")
        out.append(f"- route: **{r.get('route')}**  scores=`{r.get('route_scores')}`")
        if r.get("verdict"):
            out.append(f"- verdict: **{r['verdict']}**  (case `{r.get('case')}`, gold `{r.get('gold_answer')}`)")
        if r.get("type"):
            out.append(f"- type: `{r['type']}`  ·  elapsed: {r.get('elapsed_sec')}s")
        if r.get("route_reasoning"):
            out.append(f"- routing reasoning: {r['route_reasoning']}")
        if r.get("error"):
            out.append(f"\n> ⚠️ ERROR: `{r['error']}`")
        if r.get("cypher"):
            out.append("\n**KG Cypher (정형)**")
            out.append("```cypher\n" + r["cypher"] + "\n```")
            out.append("\n**KG context**\n```\n" + (r.get("kg_context") or "(none)") + "\n```")
        if r.get("vector_context"):
            out.append("\n**Vector 검색(top-k 청크)**\n```\n" + r["vector_context"] + "\n```")
        out.append("\n**Answer**\n")
        out.append(r.get("answer") or "(none)")
    return "\n".join(out) + "\n"


def save(results: list[dict], header: dict, outdir: str) -> tuple[Path, Path]:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    ts = header["timestamp"]
    jp = d / f"router_{ts}.json"
    mp = d / f"router_{ts}.md"
    jp.write_text(json.dumps({"header": header, "results": results},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp.write_text(build_markdown(results, header), encoding="utf-8")
    return jp, mp


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule-driven router 베이스라인(Route vs Unify)")
    ap.add_argument("question", nargs="*", help="단건 질문")
    ap.add_argument("--eval", help="jsonl 배치 실행 + 자동 채점")
    ap.add_argument("--k", type=int, default=6, help="벡터 top-k (기본 6)")
    ap.add_argument("--tau", type=float, default=0.5, help="(KG 실행기 공유 파라미터)")
    ap.add_argument("--retries", type=int, default=2, help="Cypher 재생성 횟수")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--outdir", default="results_router", help="결과 저장 폴더")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    graph = Neo4jClient()
    llm = LLMClient()
    catalog = graph.load_entity_catalog()
    kg = VanillaStructuredRetriever(graph, llm, catalog, tau=args.tau,
                                    retries=args.retries, debug=args.debug)
    vec = VectorRetriever(k=args.k)
    router = RuleRouter(llm)
    print(f"카탈로그 {len(catalog.by_id)} 엔티티 / 정형 관계타입 {len(kg.all_types)}종 / "
          f"벡터 top-k {args.k} / 라우터 4-way {PATHS}")

    header = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": "Rule-driven router (DB/Doc/Hybrid/LLM, fixed rules, arXiv:2510.02388 A_ROUTING)",
        "catalog_size": len(catalog.by_id),
        "n_structured_types": len(kg.all_types),
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
                res = run_one(router, kg, vec, llm, it.get("q", ""), it)
                show(res, args.debug)
                results.append(res)
        else:
            if not args.question:
                ap.error("질문을 입력하거나 --eval 을 쓰세요.")
            res = run_one(router, kg, vec, llm, " ".join(args.question))
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

    header["routing_summary"] = routing_summary(results)
    print_routing_summary(header["routing_summary"])
    header["system_summary"] = system_summary(results)
    print_system_summary(header["system_summary"])

    if results and not args.no_save:
        jp, mp = save(results, header, args.outdir)
        print("=" * 70)
        print(f"저장 완료:\n  {jp}\n  {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
