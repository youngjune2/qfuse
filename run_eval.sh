#!/usr/bin/env bash
# =============================================================================
# QFuse — end-to-end reproduction of the main results (Table 2 & Table 3).
#
# Prerequisites (see README):
#   - Neo4j running at $NEO4J_URI
#   - vLLM (OpenAI-compatible) generation server at $VLLM_BASE_URL  (VLLM_MODEL)
#   - embedding server at $EMB_BASE_URL                             (EMB_MODEL)
#   - pip install -r requirements.txt
# =============================================================================
set -eo pipefail

EVAL=data/eval_questions.jsonl
OUT=results
mkdir -p "$OUT"/{qfuse,hippo,light,hybrid,router}

echo "########## 1) Build structured backbone KG (Direct Mapping) ##########"
python build_graph_neo4j.py

echo "########## 2) Extract + inventory-normalize unstructured triples ##########"
python run_extract.py --normalize llm_sense

echo "########## 3) Build vector index (for HybridRAG / RouteRAG) ##########"
python vectorstore.py --build

echo "########## 4) QFuse (ours) ##########"
python run_query.py    --eval "$EVAL" --prune llm --outdir "$OUT/qfuse"

echo "########## 5) Baselines (same fused KG, retriever swapped) ##########"
python run_graphrag.py --method hippo    --eval "$EVAL" --outdir "$OUT/hippo"
python run_graphrag.py --method lightrag --eval "$EVAL" --outdir "$OUT/light"
python run_hybrid.py                     --eval "$EVAL" --outdir "$OUT/hybrid"
python run_router.py                     --eval "$EVAL" --outdir "$OUT/router"

echo "########## DONE — results in $OUT/ ##########"
