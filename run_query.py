"""
run_query.py — 3단계(검색) CLI. 결과를 파일로 저장한다.

전제: build_graph_neo4j.py (Mother Graph) + run_extract.py (Chunk/extracted 엣지) 완료.

실행:
  python run_query.py "For PRD001, which supplier ...?"
  python run_query.py --debug "..."                      # seeds/pruned/cypher 콘솔 출력
  python run_query.py --eval eval_questions_bridge100.jsonl   # 배치+자동채점(권장)
  python run_query.py --interactive
  python run_query.py --eval eval_questions_bridge100.jsonl --outdir results --no-save

저장물(기본 outdir=results):
  query_<timestamp>.json  : 전체 결과(메타/cypher/facts/context/answer) — 기계 판독용
  query_<timestamp>.md    : 사람이 읽기 좋은 리포트 — 그대로 복사해 공유하기 좋음
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from graph import Neo4jClient
from llm import LLMClient
from retrieve_d import SubgraphText2CypherRetriever, Retrieval

SYNTHESIS_RULES = """Answer rules:
- Use ONLY facts present in the retrieval results. Never invent values that are not there.
- Do not assert causation from correlation. If evidence comes from a document (an 'extracted'
  relationship), describe it as what the document states/attributes, not as established fact.
- State the provenance of each key value: which node/relationship it came from, and for
  document-derived facts, cite the source chunk id and quote the shortest relevant phrase.
- Distinguish a contracted supply relationship from a document-attributed root cause.
- If the result set is empty, say you could not find it in the graph. Do not guess.
- Closed-world negatives: if the question asks whether some entity has a property/relationship
  and the retrieved set lists the actual members but does NOT include that entity, answer "No"
  and state what IS present. Absence from a complete retrieved list is evidence of "No", not
  grounds for "cannot determine".
- Answer concisely in the same language as the question."""

# 격리(같은 그래프·검색기만 스왑) 실험용 정본 지시문. 검색 패러다임에 중립적이어야 함
# (Text2Cypher/semantic-search/PPR·A·B·C 블록 같은 검색기별 표현은 여기 넣지 않음 — 그건 각
# 시스템의 user 메시지 context 레이아웃으로만 들어간다). ours·graphrag·hybrid 이 문자열을 공유.
SYNTHESIS_SYSTEM = (
    "You are a manufacturing/supply-chain analyst answering from a single knowledge graph "
    "that fuses relational tables and company documents. The retrieval results below are the "
    "output of a retrieval step that already encodes the question's constraints, so when the "
    "results contain rows or records that answer the question, report that answer directly — do "
    "NOT reply that you could not find it while matching rows are present, and do not demand that "
    "the linking reasoning be restated in the results. Use all of the retrieved evidence "
    "(structured rows, relationships, and any document passages); a single well-supported result "
    "from one source is sufficient. Apply the yes/no and closed-world rules below.\n\n"
    + SYNTHESIS_RULES)

_SYSTEM = SYNTHESIS_SYSTEM   # 하위호환(내부 참조 유지)


def synthesize(llm: LLMClient, question: str, r: Retrieval) -> str:
    user = f"Question: {question}\n\n[Retrieval results]\n{r.context_text}"
    return llm.chat(_SYSTEM, user, max_tokens=700)


# 무정보/거부(abstention): 모델이 "찾을 수 없다/결과 없음"으로 판단 보류. 긍정도 부정도 아님.
# 극성 판정 '전에' 먼저 걸러야 함 — 이 답들은 "a yes or no answer", "is a contracted supplier"(질문 에코)
# 같은 문구를 포함해 긍정·부정 단서가 동시에 잡히기 때문.
_ABSTAIN = re.compile(
    r"cannot (determine|confirm|deny|provide|verify|say|tell)"
    r"|could ?n[o']?t (find|determine|confirm)"
    r"|no (information|data|evidence|record|results?)\b"
    r"|results? (are|is|were|was) empty"
    r"|unable to (determine|confirm|find)"
    r"|not (documented|present|available|found) in")
# 'a definitive yes or no answer' 류 잡음 — 극성 판정 전에 제거
_YESNO_NOISE = re.compile(r"(a )?(definitive |clear |simple )?[\"']?yes[\"']? or [\"']?no[\"']?( answer)?")
_AFFIRM = re.compile(r"\byes\b|is indeed|is listed|is one of the|is (a )?contracted supplier"
                     r"|confirmed (as|to be)|\bcorrect\b")
_NEGATE = re.compile(r"\bno\b|not a contracted|not contracted|is not|isn'?t|does not|does n'?t"
                     r"|no supply contract|not among|not on the|only via|only through")


def _polarity(answer: str) -> str:
    """답변의 극성: YES / NO / ABSTAIN(무정보·거부) / UNCLEAR."""
    a = (answer or "").lower()
    if _ABSTAIN.search(a):
        return "ABSTAIN"
    a = _YESNO_NOISE.sub(" ", a)          # 'yes or no' 잡음 제거 후 판정
    aff, neg = bool(_AFFIRM.search(a)), bool(_NEGATE.search(a))
    if aff and not neg:
        return "YES"
    if neg and not aff:
        return "NO"
    return "UNCLEAR"


def _grade_polarity(gold: str, answer: str, lenient: bool = True) -> str:
    """정답 NO/YES를 답변 극성으로 채점.
    트랩(NO): lenient면 '긍정만 아니면' 방어 성공(ABSTAIN/NO=PASS) — 트랩의 목적이
    '브리지 공급사를 계약사로 오인(=긍정)하는가'이므로. strict면 확정 NO만 PASS."""
    p = _polarity(answer)
    if gold.upper() == "NO":
        ok = p in ("NO", "ABSTAIN") if lenient else (p == "NO")
    else:                                  # YES
        ok = (p == "YES")
    return "PASS" if ok else "FAIL"


# 엔티티 ID -> 사람이 읽는 이름. line/region/supplier 등은 ID든 이름이든 똑같이 유일 식별하므로
# 채점에서 둘 중 하나만 답에 있어도 정답으로 인정한다(RETURN 절 비결정성으로 name-only 답이
# 억울하게 FAIL 되는 문제 방지). PO/Contract/Anomaly/Ticket/Downtime 은 이름이 없어 ID로만 채점.
_NAMED_ENTITY_FILES = [
    ("products.csv",  "product_id",   "product_name"),
    ("suppliers.csv", "supplier_id",  "supplier_name"),
    ("equipment.csv", "equipment_id", "equipment_name"),
    ("lines.csv",     "line_id",      "line_name"),
    ("regions.csv",   "region_id",    "region_name"),
    ("customers.csv", "customer_id",  "customer_name"),
]


@lru_cache(maxsize=1)
def _id_name_aliases() -> dict:
    """DATA_DIR(build_graph_neo4j 와 동일 env)에서 오프라인 로드. 실패하면 빈 맵(기존 substring 동작)."""
    data_dir = os.getenv("DATA_DIR", "data")
    out: dict[str, str] = {}
    for fn, idc, namec in _NAMED_ENTITY_FILES:
        try:
            with open(os.path.join(data_dir, fn), encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    i, n = (row.get(idc) or "").strip(), (row.get(namec) or "").strip()
                    if i and n:
                        out[i] = n
        except Exception:
            continue
    return out


def _gold_present(token: str, answer_lower: str, aliases: dict) -> bool:
    """gold 토큰이 답에 있는가 — ID 부분문자열 OR 그 ID의 엔티티 이름 부분문자열."""
    if token.lower() in answer_lower:
        return True
    nm = aliases.get(token)
    return bool(nm and nm.lower() in answer_lower)


def grade(answer: str, item: dict | None) -> str | None:
    """자동 채점. eval 아이템에 'answer'가 있을 때만 PASS/FAIL, 없으면 None(수동 판독).
      - 'NO'/'YES' (trap-negative)     : 답변 극성으로 판정.
      - 'A;B;C' (structured-only 등)    : 세미콜론 분리 후 전부 포함되면 PASS.
      - 단일 ID   (bridge-*)           : 부분문자열 포함이면 PASS.
    ID 는 대응 엔티티 이름으로도 인정(_gold_present)."""
    if not item:
        return None
    gold = item.get("answer")
    if gold is None or gold == "":
        return None
    gold = str(gold).strip()
    if gold.upper() in ("NO", "YES"):
        return _grade_polarity(gold, answer)
    golds = [g.strip() for g in gold.split(";") if g.strip()]
    a = (answer or "").lower()
    aliases = _id_name_aliases()
    return "PASS" if all(_gold_present(g, a, aliases) for g in golds) else "FAIL"


# ---------------------------------------------------------------------------
# 실행 단위
# ---------------------------------------------------------------------------
def run_one(retriever, llm, question: str, item: dict | None = None) -> dict:
    t0 = time.perf_counter()
    try:
        r = retriever.retrieve(question)
        answer = synthesize(llm, question, r)
        return {
            "question": question,
            "type": (item or {}).get("type"),
            "case": (item or {}).get("case"),
            "gold_answer": (item or {}).get("answer"),
            "verdict": grade(answer, item),
            "expect": (item or {}).get("expect"),
            "seeds": r.meta.get("seeds"),
            "n_all_types": r.meta.get("n_all_types"),
            "pruned_types": r.meta.get("pruned_types"),
            "attempts": r.meta.get("attempts"),
            "provenance_chunks": r.meta.get("provenance_chunks"),
            "tried": r.meta.get("tried"),
            "schema_text": r.meta.get("schema_text"),
            "error": r.meta.get("error"),
            "cypher": r.cypher,
            "facts": r.facts,
            "context_text": r.context_text,
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
        tag = res["case"] or ""
        print(f"    [{res['verdict']}] {tag}  (gold={res.get('gold_answer')})")
    if res.get("expect"):
        print(f"    expect: {res['expect']}")
    if res.get("error"):
        print(f"    ERROR: {res['error']}")
    if debug:
        print(f"    seeds: {res.get('seeds')}")
        pt = res.get("pruned_types") or []
        print(f"    pruned: {len(pt)}/{res.get('n_all_types')} -> {pt}")
        print(f"    provenance: {res.get('provenance_chunks')}")
        print("\n--- cypher ---")
        print(res.get("cypher") or "(none)")
        print("\n--- context ---")
        print((res.get("context_text") or "")[:2000])
    print("\n--- answer ---")
    print(res.get("answer") or "(none)")
    print()


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
def accuracy_summary(results: list[dict]) -> dict:
    """자동 채점 가능한(gold_answer 있는) 문항에 대한 집계."""
    graded = [r for r in results if r.get("verdict") in ("PASS", "FAIL")]
    n_pass = sum(1 for r in graded if r["verdict"] == "PASS")
    n_err = sum(1 for r in results if r.get("verdict") == "ERROR")
    by_type: dict[str, list[int]] = {}
    for r in graded:
        t = r.get("type") or "?"
        b = by_type.setdefault(t, [0, 0])         # [pass, total]
        b[1] += 1
        if r["verdict"] == "PASS":
            b[0] += 1
    return {"graded": len(graded), "pass": n_pass,
            "accuracy": round(n_pass / len(graded), 4) if graded else None,
            "errors": n_err, "by_type": by_type}


def system_summary(results: list[dict]) -> dict:
    """시스템 관점(운영) 지표. accuracy와 별개. D/Hybrid 공용 — 없는 키는 무시.
    latency, Cypher 시도/재시도율, 실행 에러율, 빈 결과율(D), 스키마 프루닝 비율(D)."""
    import statistics
    n = len(results)
    if not n:
        return {"n": 0}
    lat = [r["elapsed_sec"] for r in results
           if isinstance(r.get("elapsed_sec"), (int, float))]

    def _ntry(r):
        t = r.get("tried")
        if isinstance(t, list) and t:
            return len(t)
        a = r.get("attempts")
        return a if isinstance(a, int) else None
    ntry = [x for x in (_ntry(r) for r in results) if x is not None]

    errs = sum(1 for r in results
               if r.get("error") or r.get("kg_error") or r.get("verdict") == "ERROR")
    # 빈 결과율: facts 필드가 있는(=검색 실행된) 문항 중 facts==[] 비율 (D 전용)
    with_facts = [r for r in results if isinstance(r.get("facts"), list)]
    empties = sum(1 for r in with_facts if len(r["facts"]) == 0)
    # 스키마 프루닝 비율: pruned_types/n_all_types (D 전용)
    pr = [len(r["pruned_types"]) / r["n_all_types"] for r in results
          if isinstance(r.get("n_all_types"), int) and r["n_all_types"] > 0
          and isinstance(r.get("pruned_types"), list)]

    def _m(xs, f): return round(f(xs), 3) if xs else None
    p95 = round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 2) if lat else None
    return {
        "n": n,
        "latency_sec": {"mean": _m(lat, statistics.mean),
                        "median": _m(lat, statistics.median),
                        "p95": p95,
                        "total": round(sum(lat), 1) if lat else None},
        "cypher_attempts": {"mean": _m(ntry, statistics.mean),
                            "max": max(ntry) if ntry else None,
                            "retry_rate": round(sum(x > 1 for x in ntry) / len(ntry), 3) if ntry else None},
        "error_rate": round(errs / n, 3),
        "empty_result_rate": round(empties / len(with_facts), 3) if with_facts else None,
        "schema_prune_ratio_mean": _m(pr, statistics.mean),
    }


def print_system_summary(sysm: dict):
    if not sysm or not sysm.get("n"):
        return
    lat, ca = sysm.get("latency_sec", {}), sysm.get("cypher_attempts", {})
    print("=" * 70)
    print(f"[시스템] latency mean {lat.get('mean')}s / median {lat.get('median')}s / "
          f"p95 {lat.get('p95')}s / total {lat.get('total')}s")
    print(f"         cypher attempts mean {ca.get('mean')} (max {ca.get('max')}, "
          f"retry_rate {ca.get('retry_rate')}) / error_rate {sysm.get('error_rate')}")
    if sysm.get("empty_result_rate") is not None:
        print(f"         empty_result_rate {sysm.get('empty_result_rate')} / "
              f"schema_prune_ratio {sysm.get('schema_prune_ratio_mean')}")


def build_markdown(results: list[dict], header: dict) -> str:
    out = ["# KG-RAG 검색 결과 (전략 D)\n"]
    out.append(f"- 생성 시각: {header['timestamp']}")
    out.append(f"- 문항 수: {len(results)}")
    out.append(f"- 카탈로그 엔티티: {header.get('catalog_size')}")
    out.append(f"- 전역 관계타입: {header.get('n_rel_types')}")
    acc = header.get("accuracy_summary")
    if acc and acc.get("graded"):
        out.append(f"- **정확도: {acc['pass']}/{acc['graded']} = {acc['accuracy']}**"
                   f"  (ERROR {acc['errors']})")
        for t, (p, tot) in sorted(acc["by_type"].items()):
            out.append(f"    - `{t}`: {p}/{tot}")
    sysm = header.get("system_summary")
    if sysm and sysm.get("n"):
        lat, ca = sysm.get("latency_sec", {}), sysm.get("cypher_attempts", {})
        out.append(f"- **시스템**: latency mean {lat.get('mean')}s / p95 {lat.get('p95')}s"
                   f"  ·  cypher attempts mean {ca.get('mean')} (retry {ca.get('retry_rate')})"
                   f"  ·  error_rate {sysm.get('error_rate')}"
                   f"  ·  empty {sysm.get('empty_result_rate')}"
                   f"  ·  prune_ratio {sysm.get('schema_prune_ratio_mean')}")
    out.append("")

    for i, r in enumerate(results, 1):
        out.append("\n---\n")
        out.append(f"## Q{i}. {r['question']}\n")
        if r.get("verdict"):
            out.append(f"- verdict: **{r['verdict']}**  (case `{r.get('case')}`, gold `{r.get('gold_answer')}`)")
        if r.get("type"):
            out.append(f"- type: `{r['type']}`")
        if r.get("expect"):
            out.append(f"- expect: {r['expect']}")
        out.append(f"- elapsed: {r.get('elapsed_sec')}s")
        if r.get("error"):
            out.append(f"\n> ⚠️ ERROR: `{r['error']}`")
            if r.get("traceback"):
                out.append("```\n" + r["traceback"].rstrip() + "\n```")
        out.append(f"- seeds: `{r.get('seeds')}`")
        pt = r.get("pruned_types") or []
        out.append(f"- pruned types: **{len(pt)}/{r.get('n_all_types')}**")
        out.append(f"```\n{', '.join(pt)}\n```")
        out.append(f"- attempts: {r.get('attempts')}  ·  provenance: `{r.get('provenance_chunks')}`")
        out.append("\n**Cypher**")
        out.append("```cypher\n" + (r.get("cypher") or "(none)") + "\n```")
        tried = r.get("tried") or []
        if len(tried) > 1:
            out.append(f"\n**시도 {len(tried)}회 (재시도 내역)**")
            for k, t in enumerate(tried, 1):
                out.append(f"```cypher\n-- attempt {k}\n{t}\n```")
        out.append("\n**Retrieval context**")
        out.append("```\n" + (r.get("context_text") or "(none)") + "\n```")
        out.append("\n**Answer**\n")
        out.append(r.get("answer") or "(none)")
    return "\n".join(out) + "\n"


def save(results: list[dict], header: dict, outdir: str) -> tuple[Path, Path]:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    ts = header["timestamp"]
    jp = d / f"query_{ts}.json"
    mp = d / f"query_{ts}.md"
    jp.write_text(json.dumps({"header": header, "results": results},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    mp.write_text(build_markdown(results, header), encoding="utf-8")
    return jp, mp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="KG-RAG 검색 (전략 D)")
    ap.add_argument("question", nargs="*", help="질문")
    ap.add_argument("--debug", action="store_true", help="seeds/pruned/cypher/context 콘솔 출력")
    ap.add_argument("--interactive", action="store_true", help="대화형")
    ap.add_argument("--eval", help="jsonl 배치 실행 (필드: q, type, case, answer, expect). "
                                    "answer 있으면 자동 채점(정확도 요약)")
    ap.add_argument("--tau", type=float, default=0.5, help="extracted 신뢰 임계값")
    ap.add_argument("--retries", type=int, default=2, help="Cypher 재생성 횟수")
    ap.add_argument("--min-support", type=int, default=2,
                    help="extracted 관계타입이 프루닝 후보에 들어가기 위한 최소 지지 엣지 수 "
                         "(오타/1회성 추출 predicate 배제용, 기본 2)")
    ap.add_argument("--prune", choices=["llm", "none"], default="llm",
                    help="llm=LLM 관련도 프루닝(+브리지 recall 바닥, 기본) | none=프루닝 없이 전체 스키마")
    ap.add_argument("--no-dedup", action="store_true",
                    help="구조화 쌍 위 extracted 중복 제거를 끈다(비교용, 기본은 제거 on)")
    ap.add_argument("--bridge-match", choices=["typed", "untyped"], default="typed",
                    help="브리지 가운데 홉 매칭: typed=대표명 조회(기본) | "
                         "untyped=이름 무관 -[r]- WHERE origin='extracted'(정규화 오류·파편화 면역, 방법2)")
    ap.add_argument("--outdir", default="results", help="결과 저장 폴더 (기본: results)")
    ap.add_argument("--no-save", action="store_true", help="파일 저장 없이 콘솔만")
    args = ap.parse_args()

    graph = Neo4jClient()
    llm = LLMClient()
    catalog = graph.load_entity_catalog()
    rel_types = graph.relationship_types()
    print(f"카탈로그 {len(catalog.by_id)} 엔티티 / 관계타입 {len(rel_types)}종")

    retriever = SubgraphText2CypherRetriever(
        graph, llm, catalog, tau=args.tau, retries=args.retries, debug=args.debug,
        min_extracted_support=args.min_support, prune_mode=args.prune,
        dedup_structured=not args.no_dedup, bridge_match=args.bridge_match)

    header = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "catalog_size": len(catalog.by_id),
        "n_rel_types": len(rel_types),
        "relationship_types": rel_types,
        "tau": args.tau,
        "min_support": args.min_support,
        "prune_mode": args.prune,
        "dedup_structured": not args.no_dedup,
        "bridge_match": args.bridge_match,
        "dropped_low_support_types": retriever.dropped_low_support,
    }
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
        elif args.interactive:
            print("대화형 모드. 'exit' 종료.")
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() in ("exit", "quit"):
                    break
                res = run_one(retriever, llm, q)
                show(res, args.debug)
                results.append(res)
        else:
            if not args.question:
                ap.error("질문을 입력하거나 --interactive / --eval 을 쓰세요.")
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