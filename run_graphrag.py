"""
run_graphrag.py — 그래프-RAG 검색 베이스라인(HippoRAG / LightRAG) 실행·평가.

우리 방법(retrieve_d)이 쓰는 것과 **동일한 살아있는 B2 fused 그래프** 위에서
HippoRAG(PPR) 또는 LightRAG(dual-level 키워드) 검색을 돌린다. 세 방법의 유일한 차이가
'검색 알고리즘'이 되도록 격리한 공정 비교(자세한 원칙은 retrieve_graphrag 참조).

Strategy D / HybridRAG / Router 와 동일한 eval 파일·채점기(run_query.grade)·system_summary 를
써서 그대로 한 표에 비교할 수 있다.

전제:
  1) build_graph_neo4j.py + run_extract.py 완료(B2 그래프가 Neo4j 에 있음)
  2) 임베딩 엔드포인트(:8001) 가동 — LightRAG 필수, HippoRAG 는 NER 이름링킹에만 사용(선택).

실행:
  python run_graphrag.py --method hippo    --eval eval_questions_bridge100.jsonl
  python run_graphrag.py --method lightrag --eval eval_questions_bridge100.jsonl
  python run_graphrag.py --method hippo --debug "For downtime DT0007, which supplier ...?"
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
from retrieve_graphrag import GraphData, EntityEmbeddingIndex, graphrag_synthesize
from retrieve_hippo import HippoRetriever
from retrieve_lightrag import LightRAGRetriever
from run_query import grade, accuracy_summary, system_summary, print_system_summary


def run_one(retriever, llm, question: str, item: dict | None = None) -> dict:
    t0 = time.perf_counter()
    try:
        r = retriever.retrieve(question)
        answer = graphrag_synthesize(llm, question, r["ent_ctx"], r["rel_ctx"], r["psg_ctx"])
        return {
            "question": question,
            "type": (item or {}).get("type"),
            "case": (item or {}).get("case"),
            "gold_answer": (item or {}).get("answer"),
            "verdict": grade(answer, item),
            "expect": (item or {}).get("expect"),
            "seeds": r.get("seeds"),
            "entities": r.get("entities"),
            "passages": r.get("passages"),
            # empty_result_rate(system_summary)용: 회수된 근거(엔티티+구절)
            "facts": (r.get("entities") or []) + (r.get("passages") or []),
            "ent_context": r["ent_ctx"],
            "rel_context": r["rel_ctx"],
            "psg_context": r["psg_ctx"],
            "retrieval_meta": r.get("meta"),
            "low_keywords": r.get("low_keywords"),
            "high_keywords": r.get("high_keywords"),
            "answer": answer,
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
        if res.get("low_keywords") is not None:
            print(f"    keywords low={res.get('low_keywords')} high={res.get('high_keywords')}")
        print(f"    retrieval: {res.get('retrieval_meta')}")
        print("\n--- entities ---")
        print((res.get("ent_context") or "")[:1200])
        print("\n--- relationships ---")
        print((res.get("rel_context") or "")[:1200])
        print("\n--- passages ---")
        print((res.get("psg_context") or "")[:1500])
    print("\n--- answer ---")
    print(res.get("answer") or "(none)")
    print()


def build_markdown(results: list[dict], header: dict) -> str:
    out = [f"# 그래프-RAG 베이스라인 결과 — {header.get('method')}\n"]
    out.append(f"- 생성 시각: {header['timestamp']}")
    out.append(f"- 문항 수: {len(results)}")
    out.append(f"- 그래프: 노드 {header.get('n_nodes')} / 엣지 {header.get('n_edges')} / 청크 {header.get('n_chunks')}")
    out.append(f"- 파라미터: {header.get('params')}")
    acc = header.get("accuracy_summary")
    if acc and acc.get("graded"):
        out.append(f"- **정확도: {acc['pass']}/{acc['graded']} = {acc['accuracy']}**  (ERROR {acc['errors']})")
        for t, (p, tot) in sorted(acc["by_type"].items()):
            out.append(f"    - `{t}`: {p}/{tot}")
    sysm = header.get("system_summary")
    if sysm and sysm.get("n"):
        lat = sysm.get("latency_sec", {})
        out.append(f"- **시스템**: latency mean {lat.get('mean')}s / p95 {lat.get('p95')}s"
                   f"  ·  error_rate {sysm.get('error_rate')}  ·  empty {sysm.get('empty_result_rate')}")
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
            if r.get("traceback"):
                out.append("```\n" + r["traceback"].rstrip() + "\n```")
        out.append(f"- seeds: `{r.get('seeds')}`  ·  retrieval: `{r.get('retrieval_meta')}`")
        if r.get("low_keywords") is not None:
            out.append(f"- keywords: low `{r.get('low_keywords')}` / high `{r.get('high_keywords')}`")
        out.append("\n**Entities**")
        out.append("```\n" + (r.get("ent_context") or "(none)") + "\n```")
        out.append("\n**Relationships**")
        out.append("```\n" + (r.get("rel_context") or "(none)") + "\n```")
        out.append("\n**Passages**")
        out.append("```\n" + (r.get("psg_context") or "(none)") + "\n```")
        out.append("\n**Answer**\n")
        out.append(r.get("answer") or "(none)")
    return "\n".join(out) + "\n"


def save(results: list[dict], header: dict, outdir: str) -> tuple[Path, Path]:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    ts = header["timestamp"]
    tag = header.get("method", "graphrag")
    jp = d / f"{tag}_{ts}.json"
    mp = d / f"{tag}_{ts}.md"
    jp.write_text(json.dumps({"header": header, "results": results},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp.write_text(build_markdown(results, header), encoding="utf-8")
    return jp, mp


def main() -> int:
    ap = argparse.ArgumentParser(description="그래프-RAG 베이스라인(HippoRAG / LightRAG)")
    ap.add_argument("question", nargs="*", help="단건 질문")
    ap.add_argument("--method", required=True, choices=["hippo", "lightrag"], help="검색 방법")
    ap.add_argument("--eval", help="jsonl 배치 실행 + 자동 채점")
    ap.add_argument("--tau", type=float, default=0.5, help="extracted 신뢰 임계값(그래프 로딩 공유)")
    ap.add_argument("--k-passages", type=int, default=12, help="HippoRAG top-k passage")
    ap.add_argument("--m-entities", type=int, default=15, help="HippoRAG top-m 엔티티")
    ap.add_argument("--alpha", type=float, default=0.15, help="PPR teleport 확률")
    ap.add_argument("--link-tau", type=float, default=0.5,
                    help="HippoRAG NER 임베딩-링킹 코사인 floor")
    ap.add_argument("--link-margin", type=float, default=0.05,
                    help="HippoRAG NER 링킹 모호성 게이트: top1-top2 코사인 차가 이 값 이상일 때만 채택")
    ap.add_argument("--k-low", type=int, default=5, help="LightRAG low 키워드당 엔티티 수")
    ap.add_argument("--k-type", type=int, default=3, help="LightRAG high 키워드당 관계타입 수")
    ap.add_argument("--r-max", type=int, default=30, help="LightRAG 관계 상한")
    ap.add_argument("--ent-max", type=int, default=30, help="LightRAG 엔티티 상한")
    ap.add_argument("--chunk-max", type=int, default=8, help="LightRAG 청크 상한")
    ap.add_argument("--no-emb", action="store_true", help="임베딩 인덱스 생략(HippoRAG 정확매칭만)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--outdir", default="results_graphrag", help="결과 저장 폴더")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    graph = Neo4jClient()
    llm = LLMClient()
    catalog = graph.load_entity_catalog()
    gd = GraphData(graph, tau=args.tau, debug=True)

    emb_index = None
    if not args.no_emb:
        try:
            emb_index = EntityEmbeddingIndex(gd, debug=True)
        except Exception as e:
            if args.method == "lightrag":
                graph.close()
                print(f"[fatal] LightRAG 는 임베딩 인덱스가 필요합니다: {type(e).__name__}: {e}")
                print("       :8001 임베딩 서버(qwen3-embedding)를 확인하세요.")
                return 2
            print(f"[warn] 임베딩 인덱스 실패({type(e).__name__}) -> HippoRAG 정확매칭 링킹만 사용")

    if args.method == "hippo":
        retriever = HippoRetriever(gd, llm, catalog, emb_index=emb_index,
                                   k_passages=args.k_passages, m_entities=args.m_entities,
                                   alpha=args.alpha, link_tau=args.link_tau,
                                   link_margin=args.link_margin, debug=args.debug)
        params = {"k_passages": args.k_passages, "m_entities": args.m_entities,
                  "alpha": args.alpha, "link_tau": args.link_tau,
                  "link_margin": args.link_margin, "emb_linking": emb_index is not None}
    else:
        retriever = LightRAGRetriever(gd, llm, catalog, emb_index=emb_index,
                                      k_low=args.k_low, k_type=args.k_type, r_max=args.r_max,
                                      ent_max=args.ent_max, chunk_max=args.chunk_max,
                                      link_tau=args.link_tau, link_margin=args.link_margin,
                                      debug=args.debug)
        params = {"k_low": args.k_low, "k_type": args.k_type, "r_max": args.r_max,
                  "ent_max": args.ent_max, "chunk_max": args.chunk_max,
                  "link_tau": args.link_tau, "link_margin": args.link_margin}

    header = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": args.method,
        "graph": "same live B2 fused graph (retrieve_d 와 동일)",
        "n_nodes": len(gd.nodes), "n_edges": len(gd.edges), "n_chunks": len(gd.chunks),
        "catalog_size": len(catalog.by_id),
        "tau": args.tau,
        "params": params,
    }
    print(f"카탈로그 {len(catalog.by_id)} 엔티티 / method={args.method} / params={params}")

    results: list[dict] = []
    try:
        if args.eval:
            items = [json.loads(l) for l in
                     Path(args.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, it in enumerate(items, 1):
                print("=" * 70)
                print(f"Q{i}/{len(items)} [{it.get('type','')}]")
                res = run_one(retriever, llm, it.get("q", ""), it)
                show(res, args.debug)
                results.append(res)
        else:
            if not args.question:
                ap.error("질문을 입력하거나 --eval 을 쓰세요.")
            res = run_one(retriever, llm, " ".join(args.question))
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
