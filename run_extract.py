"""
run_extract.py — 2단계(추출) 러너.  카탈로그 소스 = CSV.
순서: (Document)-[:HAS_CHUNK]->(Chunk) 적재 + full-text 인덱스
      -> 전체 문서 추출(LLM) -> confidence 승격 -> origin='extracted' 엣지 write.

전제:
  - build_graph_neo4j.py 로 Mother Graph 적재 완료.
  - vLLM(OpenAI 호환) 기동.

실행:
  python run_extract.py            # 전체 문서 추출 + 적재
  python run_extract.py --dry      # 쓰기 없이 승격 결과만 출력
  python run_extract.py --limit 5  # 앞 5개 문서만 (빠른 확인용)
"""

from __future__ import annotations

import os
import sys
import argparse

import pandas as pd
from neo4j import GraphDatabase

import extract as EX
import confidence as C
from llm import LLMClient, EmbeddingClient   # 공용 LLM + 임베딩(정규화 방법2) 클라이언트

# --- 설정 (env로 덮어쓰기 가능) ---
URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DATA_DIR = os.getenv("DATA_DIR", "data")
DOCUMENTS_FILE = os.getenv("DOCUMENTS_FILE", "documents.csv")   # 교체: documents.csv -> _new


def main():
    ap = argparse.ArgumentParser(description="KG-RAG 2단계 추출 러너")
    ap.add_argument("--dry", action="store_true", help="쓰기 없이 승격 결과만 출력")
    ap.add_argument("--limit", type=int, default=None, help="앞 N개 문서만 처리")
    ap.add_argument("--tau", type=float, default=0.5, help="승격 임계값")
    ap.add_argument("--show", type=int, default=20, help="출력할 승격 엣지 수(0=전체)")
    ap.add_argument("--normalize",
                    choices=["off", "endpoint_override", "embed_cluster", "guarded_override", "llm_sense"],
                    default="guarded_override", help="predicate 파편화 정규화 모드(추출 후 배치, 기본 guarded). "
                         "llm_sense=아는 브리지쌍은 chat-LLM이 evidence 읽고 sense 분류(방법1)")
    ap.add_argument("--emb-tau", type=float, default=0.82,
                    help="embed_cluster: 코사인 병합 임계")
    ap.add_argument("--sense-tau", type=float, default=0.55,
                    help="guarded_override: sense 프로토타입 코사인 이 값 미만이면 매핑 거부(원본 유지)")
    ap.add_argument("--sense-margin", type=float, default=0.0,
                    help="guarded_override: top1-top2 이 값 미만이면 애매로 보고 거부(0=off)")
    # --- Entity 구속 축(ablation) ---
    ap.add_argument("--entities", choices=["bound", "open"], default="bound",
                    help="bound=백본 해소 엔티티만(구속, 현재). open=비-백본 추출 엔티티도 :ExtractedEntity 로 유지")
    ap.add_argument("--entity-merge", choices=["string", "embed"], default="string",
                    help="open 모드 비-백본 엔티티 병합: string(정준키·결정론) | embed(임베딩 τ)")
    ap.add_argument("--entity-tau", type=float, default=0.86,
                    help="entity-merge=embed: 코사인 이 값 이상이면 같은 엔티티로 병합")
    args = ap.parse_args()

    docs = pd.read_csv(os.path.join(DATA_DIR, DOCUMENTS_FILE),
                       encoding="utf-8-sig").to_dict("records")
    if args.limit:
        docs = docs[: args.limit]
    cat = EX.Catalog.from_csv(DATA_DIR)
    cfg = EX.ExtractConfig(conf=C.ConfidenceConfig(tau=args.tau),
                           normalize_mode=args.normalize, emb_tau=args.emb_tau,
                           sense_tau=args.sense_tau, sense_margin=args.sense_margin,
                           entities=args.entities, entity_merge=args.entity_merge,
                           entity_tau=args.entity_tau)
    print(f"문서 {len(docs)} / 카탈로그 {len(cat.rows)} 엔티티 / "
          f"win={cfg.window_size},step={cfg.window_step},tau={cfg.conf.tau} / "
          f"normalize={cfg.normalize_mode}")

    llm = LLMClient()
    _need_emb = args.normalize in ("embed_cluster", "guarded_override") or \
        (args.entities == "open" and args.entity_merge == "embed")
    emb = EmbeddingClient() if _need_emb else None
    ops, breakdowns, funnel = EX.run_extraction(docs, cat, llm, cfg, emb)
    print(f"추출 트리플 {len(breakdowns)} -> 승격 {len(ops)} "
          f"(탈락 {len(breakdowns) - len(ops)})")

    # --- 선택 깔때기 계측: 순수 로깅(그래프/결과 불변). 소실이 어디서 얼마나 나는지 기록. ---
    fd = funnel.to_dict()
    print("\n선택 깔때기 (loss accounting)")
    print(f"  raw LLM triples          {fd['raw_llm_triples']}")
    print(f"    drop empty/selfloop      {fd['drop_empty_or_selfloop']}")
    print(f"    drop entity-unresolved   {fd['drop_entity_unresolved']}  "
          f"({fd['entity_unresolved_rate']*100:.1f}%)  <- 소실")
    print(f"    drop canonical-none      {fd['drop_canonical_none']}")
    print(f"  resolved                 {fd['resolved_triples']}")
    print(f"    promoted (tau={cfg.conf.tau})       {len(ops)}")
    print(f"  predicate distinct       {fd['predicates_before_norm']} -> {fd['predicates_after_norm']} (정규화 전->후)")
    if fd["unresolved_surface_top"]:
        sample = ", ".join(f'"{s}"×{c}' for s, c in fd["unresolved_surface_top"][:12])
        print(f"  소실 surface(top): {sample}")

    _gb, _fk = fd["genuine_bridge_edges"], fd["fk_redundant_edges"]
    _tot = _gb + _fk or 1
    print("\nextracted 엣지 분할 (가치 vs 노이즈)")
    print(f"  genuine-bridge (정형쌍 없음)  {_gb}엣지 / {fd['genuine_bridge_types']}종  "
          f"({_gb/_tot*100:.1f}%)  <- 테이블로 표현 불가한 연결 = C1 가치")
    print(f"  FK-중복 (정형쌍 있음)         {_fk}엣지  ({_fk/_tot*100:.1f}%)  <- 동적 FK-fold로 정형명 흡수")

    print("\npredicate 파편화 감사")
    print(f"  distinct 총            {fd['predicate_distinct_total']}  (singleton {fd['predicate_singletons']})")
    print(f"    인벤토리(대표 sense)   {fd['inventory_distinct']}")
    print(f"    raw 잔존             {fd['residual_distinct']}종 / {fd['residual_triples']}엣지  "
          f"({fd['residual_rate']*100:.1f}%)  <- 2단 정규화 대상")
    if fd["pair_residual"]:
        print("  label쌍별 raw 잔존(합칠 후보, 잔존 2종 이상):")
        for pair, preds in list(fd["pair_residual"].items())[:10]:
            names = ", ".join(f"{p}×{c}" for p, c in preds[:6])
            print(f"    {pair:<34} [{len(preds)}종] {names}")

    os.makedirs("logs", exist_ok=True)   # 그래프가 아니라 로그 → dry 여부와 무관하게 항상 기록
    import json as _json
    with open("logs/extraction_funnel.json", "w", encoding="utf-8") as f:
        _json.dump(fd, f, ensure_ascii=False, indent=2)
    # 소실 트리플 격리 보관(버리지 않고 챙김) — 다른 데이터셋 추가 시 재검토용
    with open("logs/unresolved_triples.jsonl", "w", encoding="utf-8") as f:
        for rec in funnel.unresolved_records:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> logs/extraction_funnel.json + logs/unresolved_triples.jsonl "
          f"({len(funnel.unresolved_records)}건 격리)")

    dist: dict[str, int] = {}
    for op in ops:
        dist[op.rel_type] = dist.get(op.rel_type, 0) + 1
    for rt, c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {rt:<32} {c}")

    shown = ops if args.show == 0 else ops[: args.show]
    if args.dry:
        print(f"\n[--dry] 쓰기 생략. 승격 엣지 {len(shown)}/{len(ops)}개:")
        for op in shown:
            print(f"  ({op.src_label} {op.src_id})-[:{op.rel_type} "
                  f"conf={op.props['confidence']} src={op.props['source_chunk_id']}]"
                  f"->({op.tgt_label} {op.tgt_id})")
        return 0

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as drv:
        drv.verify_connectivity()
        n_chunks = EX.create_chunk_graph(drv, DATABASE, docs, cfg)
        print(f"chunk 그래프: Chunk {n_chunks}개 + HAS_CHUNK + full-text 인덱스(chunkText)")
        n = EX.write_ops(drv, DATABASE, ops)
        print(f"write 완료: 승격 op {n}건 적재(MERGE)")
        # --- 무결성 계측: write 후 실그래프 origin 분포. FK-fold이 정형 provenance를 덮지 않았는지 확인.
        orows, _, _ = drv.execute_query(
            "MATCH ()-[r]->() WHERE r.origin IS NOT NULL "
            "RETURN r.origin AS o, count(*) AS c ORDER BY o", database_=DATABASE)
        crows, _, _ = drv.execute_query(
            "MATCH ()-[r]->() WHERE r.doc_corroborations IS NOT NULL "
            "RETURN count(*) AS c, sum(r.doc_corroborations) AS s", database_=DATABASE)
        print("그래프 origin 분포 (write 후):")
        for row in orows:
            print(f"    origin={row['o'] or '-':<13} {row['c']}")
        print(f"    정형 엣지에 문서 corroboration 표시: {crows[0]['c']}건 (총 {crows[0]['s']}회)")
        print(f"    => origin='extracted' = genuine-bridge 근사(정형 대응 없는 doc-only 연결)")
    print("\n2단계(추출) 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
