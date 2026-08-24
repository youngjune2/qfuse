# QFuse

**Preserving Precise Queryability in Knowledge Graphs over Structured and Unstructured Data**

QFuse is a KG-RAG framework that fuses relational tables and free-text documents into a
single knowledge graph while keeping the graph schema **bounded**, so that queries can be
answered by **precise graph search** (schema-bounded Text-to-Cypher) rather than approximate
neighborhood expansion. The underlying method is an **Inventory-Bounded Knowledge Graph
(IBKG)**: a structured backbone constrains which entities and relations the unstructured side
may introduce, keeping the fused schema small enough for reliable Cypher generation.

Three bounding mechanisms:
1. **Entity Bound** — unstructured entities are admitted only if they map to the structured backbone's Entity Inventory.
2. **Relation Bound** — extracted predicates are normalized to a per-label-pair Relation Inventory.
3. **Schema Pruning** — only the query-relevant schema subset is passed to Text-to-Cypher.

This repository contains the code and the synthetic benchmark to reproduce the main results.

---

## Requirements

- **Python 3.10+**, then `pip install -r requirements.txt`
- Three services running locally:
  - **Neo4j** (graph store)
  - **vLLM** OpenAI-compatible generation server (used for extraction, normalization, Text-to-Cypher, and answer synthesis)
  - an **embedding** server (document chunk embeddings; also OpenAI-compatible)

Figure scripts (`fig_*.py`) additionally require `SciencePlots` and a LaTeX installation.

### Environment variables (defaults shown)

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | `bolt://localhost:7687` / `neo4j` / `password123` / `neo4j` | Neo4j connection |
| `VLLM_BASE_URL` / `VLLM_API_KEY` / `VLLM_MODEL` | `http://localhost:8000/v1` / `EMPTY` / `t2c-gemma3-27b` | generation LLM (shared by all roles) |
| `EMB_BASE_URL` / `EMB_API_KEY` / `EMB_MODEL` | `http://localhost:8001/v1` / `EMPTY` / `qwen3-embedding` | embedding model |
| `LLM_TEMPERATURE` | `0.0` | deterministic decoding |
| `DATA_DIR` | `data` | dataset directory |

The generation model is Text-to-Cypher–specialized (`T2C-Gemma3-27B` in the paper), served
under whatever name you pass via `VLLM_MODEL`.

---

## Quickstart

```bash
pip install -r requirements.txt
# start Neo4j, the vLLM generation server, and the embedding server, then:
bash run_eval.sh
```

`run_eval.sh` runs the full pipeline end to end: build the backbone KG, extract and
inventory-normalize the document triples, build the vector index, and evaluate QFuse plus all
four baselines into `results/`. To run stages individually:

```bash
python build_graph_neo4j.py                                   # structured -> KG (Direct Mapping)
python run_extract.py --normalize llm_sense                   # documents -> triples -> bounded merge
python vectorstore.py --build                                 # vector index (HybridRAG / RouteRAG)

python run_query.py    --eval data/eval_questions.jsonl --prune llm --outdir results/qfuse
python run_graphrag.py --method hippo    --eval data/eval_questions.jsonl --outdir results/hippo
python run_graphrag.py --method lightrag --eval data/eval_questions.jsonl --outdir results/light
python run_hybrid.py                     --eval data/eval_questions.jsonl --outdir results/hybrid
python run_router.py                     --eval data/eval_questions.jsonl --outdir results/router
```

---

## Dataset (`data/`)

A synthetic manufacturing / supply-chain dataset modeled on Walmart M5 (fully synthetic; no
real data). Some key relations exist **only** in the documents, so bridge questions require
crossing between the two sources.

- **Structured:** 13 relational tables (products, suppliers, customers, regions, lines,
  equipment, contracts, purchase orders/lines, downtime/anomaly events, service tickets)
  linked by foreign keys.
- **Unstructured:** 196 documents (`documents.csv`) with evidence links (`document_links.csv`).
- **Evaluation:** 171 questions (`eval_questions.jsonl`) — 90 Bridge (structured–unstructured–
  structured 3-hop), 30 Structured, 30 Document, 21 Trap. Each question carries its gold
  answer, which is used for grading.

Agent prompts, baseline configurations, and the full dataset card are in
[`docs/APPENDIX.md`](docs/APPENDIX.md).

Grading is accuracy: a prediction is correct if the gold answer appears in the generated
answer text (all parts for multi-answer; polarity for Trap yes/no questions).

---

## Repository layout

| Path | Description |
|---|---|
| `build_graph_neo4j.py`, `run_extract.py`, `extract.py`, `chunk.py` | offline: KG construction, triple extraction, inventory normalization |
| `retrieve_d.py`, `run_query.py` | **QFuse** retriever (bounded Text-to-Cypher) and evaluation entry point |
| `retrieve_{hippo,lightrag,graphrag,hybrid}.py`, `run_{graphrag,hybrid,router}.py`, `route.py` | baseline retrievers and runners |
| `graph.py`, `schema.py`, `llm.py`, `confidence.py`, `vectorstore.py` | shared utilities |
| `fig_ladder.py`, `fig_types.py`, `fig_ctx_c.py` | paper figures |
| `data/` | synthetic dataset (CC BY 4.0) |
| `docs/APPENDIX.md` | agent prompts, baseline configs, dataset card |

---

## License

Code is released under the **MIT License** (`LICENSE`); the dataset under `data/` is released
under **CC BY 4.0** (`data/LICENSE`).

## Citation

```bibtex
@inproceedings{kang2026qfuse,
  title     = {QFuse: Preserving Precise Queryability in Knowledge Graphs over Structured and Unstructured Data},
  author    = {Kang, Youngjune and Yoon, Jaehyung and Youn, Chae Eun and Kim, Kyungeun and Kim, Youngjae},
  year      = {2026}
}
```
