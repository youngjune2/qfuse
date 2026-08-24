# Appendix

All generation roles (extraction, normalization, retrieval, synthesis) are performed by a
**single shared LLM** (`T2C-Gemma3-27B`, specialized for Text2Cypher); embeddings use
`Qwen3-Embedding-0.6B`. Every call is run at `temperature=0` (deterministic), and steps that
require structured output use JSON-schema-constrained decoding (one retry on failure). Because
Gemma-family chat templates have no system role, the system instruction is prepended to the
user message. The prompts below are quoted verbatim from the implementation.

---

## A. Agent Prompts (IBKG)

IBKG comprises five agents — Offline: (1) Triple Extract, (2) Normalization; Online:
(3) Schema Decision (Pruning), (4) Query Build (Text2Cypher), (5) Answer.

### A.1 Triple Extract Agent

Extracts **only relationships explicitly stated between candidate entities** that exist on the
structured backbone (Entity Bound). Relationships to entities absent from the candidate set are
discarded.

```
You extract relationships that are EXPLICITLY stated in the given text, only between the
listed candidate entities. Use the exact candidate IDs for subject_id and object_id.
'predicate' is a short verb phrase describing the relation. 'evidence' is the exact span
from the text that states it. Do NOT invent relationships not present in the text.
If there is no explicit relationship, return an empty list. Output JSON only.
```

Output schema (JSON):
```json
{"triples": [{"subject_id": "...", "predicate": "...", "object_id": "...", "evidence": "..."}]}
```

> In the open condition (Entity Bound disabled, used for ablation) a variant prompt
> (`_SYSTEM_OPEN`) additionally allows non-candidate entities, using their surface phrase as id.

### A.2 Normalization Agent (Predicate Inventory Bound)

Folds each extracted relation into one representative sense from the **per-node-pair (Label
pair) Predicate Inventory**. The inventory consists of (i) items auto-derived from the FK
relations of the structured backbone and (ii) a small set of hand-authored senses for bridge
pairs. The agent reads only the evidence sentence and classifies it into exactly one sense
(or `OTHER` when none fits). The key is disambiguating pairs whose direction/causality is
opposite (e.g. a PO expedited to fix a fault vs. a defective delivery that caused the fault).

```
You label ONE relationship between two manufacturing/supply-chain entities into exactly one
of the given sense categories, judging ONLY from the evidence sentence. Read it carefully —
direction and cause/effect matter (e.g. 'a PO expedited to fix an anomaly' is the OPPOSITE of
'a defective delivery that caused the anomaly'). If none of the categories clearly fits,
answer OTHER. Output JSON only: {"sense": "<CATEGORY_NAME or OTHER>"}.
```

**Bridge-pair inventory (hand-authored, 6 pairs).** Sense (canonical name) and discriminating
gloss for each pair:

| # | Label pair | Sense (canonical) | Meaning |
|---|---|---|---|
| BT1 | DowntimeEvent–PurchaseOrder | `TRIGGERED_REPLACEMENT_ORDER` | A replacement/substitute part was ordered to recover from the downtime |
| | | `LATE_DELIVERY_CAUSED_DOWNTIME` (trap) | The order's own late/missed delivery starved the line and caused the downtime |
| BT2 | DowntimeEvent–Contract | `DELAYED_CONTRACT_DELIVERY` | The downtime delayed/breached the volume committed under the contract |
| BT3 | PurchaseOrder–AnomalyEvent | `ORDERED_TO_REMEDY_ANOMALY` | The order was placed after the anomaly to correct/source the fix |
| | | `DELIVERY_CAUSED_ANOMALY` (trap) | A defective delivery under the order was itself the cause of the anomaly |
| BT4 | ServiceTicket–PurchaseOrder | `REPLACEMENT_SOURCED_UNDER_ORDER` | The replacement unit that resolved the ticket was sourced under the PO |
| BT5 | Contract–Product | `COMMITS_PRODUCT_VOLUME` | The contract commits a fixed annual volume of this product |
| | | `EXCLUDES_PRODUCT` (trap) | The product is explicitly carved out / excluded from the commitment |
| BT6 | ServiceTicket–AnomalyEvent | `ROOT_CAUSED_TO_ANOMALY` | The ticket was escalated to engineering and root-caused to the anomaly |

The inventory for structured FK pairs is generated automatically from the build spec
(`REL_SPECS`), so it generalizes to a new dataset without hand extension; only the bridge
pairs above are hand-authored. Sense assignment is performed by the LLM from the evidence
sentence, and a relation is left unchanged when no inventory sense clearly fits (`OTHER`).

### A.3 Schema Decision (Pruning) Agent

Before Text2Cypher, retains **only the relationship types needed for the answer path** out of
the full relation schema of the Unified KG. It is instructed to include every hop of the path,
to include when unsure (recall-first), and to prefer the higher-support type (edge count n)
among synonymous relations.

```
You are selecting which graph relationships are needed to answer a question.
You are given lines of the form (SourceLabel)-[:REL_TYPE]->(TargetLabel)  [n=<edge count>].
Return the REL_TYPE names that could lie on a path from the question's entities to the answer.
Include every step of the path, not just the final hop. When unsure, include it.
If several REL_TYPEs look like they express the same idea, prefer the one(s) with the larger n
unless the question's own wording clearly matches a specific lower-n type.
Return JSON {"relationship_types": ["A","B"]} using only REL_TYPE names from the list.
```

The extracted types connecting the six known bridge Label pairs are always kept in the schema
regardless of the normalization mode, to guarantee pruning recall.

### A.4 Query Build (Text2Cypher) Agent

Given only the pruned bounded schema, generates **one read-only Cypher query**. Labels and
relationship types must be copied character-for-character from the schema; when a concept is
expressed by both, the `structured` relation is preferred, and extracted relations are gated by
`WHERE r.confidence >= τ` (τ=0.5). On generation/validation failure it retries up to twice.

```
You write ONE read-only Neo4j Cypher query answering the question, using ONLY the schema given.
Hard requirements:
- Read-only: MATCH / OPTIONAL MATCH / WHERE / RETURN / ORDER BY / LIMIT only.
- Use ONLY relationship types and node labels that appear in the schema. Never invent one.
- Relationship types and node labels must be copied CHARACTER-FOR-CHARACTER from the schema
  block — same letters, same underscores, same case. Do not paraphrase, abbreviate, remove
  underscores, or merge/duplicate letters (e.g. write SUPPLIED_BY exactly as shown; never
  SUPPLIEDBY, Suppliedby, or SUPPLIED_BYY). Before writing each [:TYPE] or (:Label), find that
  exact string in the schema block above and copy it — do not type it from memory.
- When a 'structured' and an 'extracted' relationship express the same idea, PREFER the
  'structured' one (it is complete; extracted ones cover only documents that mention it).
- Copy relationship direction exactly as written in the schema: (A)-[:REL]->(B).
- Relationship syntax: -[:REL]-> , <-[:REL]- , or -[:REL]- (undirected). Never write <-[:REL]->
  and never put a relationship type inside a node's {...} braces.
- Follow the whole path step by step; a question may need 2-3 hops through intermediate nodes.
- RETURN aliases must be single words (use AS anomaly_id, never AS Anomaly ID).
- When you traverse an 'extracted' relationship, bind it (e.g. -[r:REL]->), add
  WHERE r.confidence >= {tau}, and RETURN r.source_chunk_id AS source_chunk_id.
- Do NOT add date/time WHERE filters. The named entity IDs in the question already pin down
  the exact records, so any month or year mentioned in the question is just context, not a
  filter you must encode. An unnecessary date filter almost always returns zero rows — omit it.
- Never call a date/time constructor function such as temporal(...), date(...), datetime(...),
  or timestamp(...). temporal(...) is not a real function and raises "Unknown function 'temporal'";
  the others are unnecessary here. If — and only if — a time filter is truly unavoidable, read
  parts of the existing property directly with .year/.month/.day accessors, e.g.
  `d.start_time.year = 2026 AND d.start_time.month = 3`. Never compare a date property to a
  string literal like `d.start_time >= '2026-03-01'` (it silently matches nothing).
- Output exactly one JSON object: {"cypher": "..."}
```
(`{tau}` = 0.5.) The following few-shot examples are supplied with the prompt to fix the
schema syntax:

```
Examples of correct SYNTAX for this graph. These use different entities/relations
than the actual question below — do not copy them as if they contained the answer. Adapt the
PATTERN (hop style, WHERE/RETURN shape) to whatever entities and relationships the real
question and schema actually need.

  Q: Which purchase orders were placed with supplier SUP005, and what are their statuses?
  A: MATCH (po:PurchaseOrder)-[:FROM_SUPPLIER]->(s:Supplier {supplier_id: 'SUP005'})
     RETURN po.purchase_order_id AS purchase_order_id, po.status AS status, po.order_date AS order_date

  Q: Which purchase orders ordered product PRD005, and which supplier was each from?
  A: MATCH (po:PurchaseOrder)-[:ORDERS_PRODUCT]->(p:Product {product_id: 'PRD005'})
     MATCH (po)-[:FROM_SUPPLIER]->(s:Supplier)
     RETURN po.purchase_order_id AS purchase_order_id, s.supplier_id AS supplier_id,
            s.supplier_name AS supplier_name

  Q: Does any document attribute equipment EQ008's parts to a specific supplier?
  A: MATCH (e:Equipment {equipment_id: 'EQ008'})-[r:SUPPLIED_BY]->(s:Supplier)
     WHERE r.confidence >= 0.5
     RETURN DISTINCT s.supplier_id AS supplier_id, s.supplier_name AS supplier_name,
            r.source_chunk_id AS source_chunk_id

  Note: always write property access as variable.property (st.ticket_id), never st_ticket_id.
  Use DISTINCT when a pattern can match the same node many times. When a question needs a
  multi-hop causal chain (e.g. product -> downtime -> anomaly -> equipment/supplier), walk it
  hop by hop using the relationships that actually exist in the schema above — do not assume a
  direct shortcut relationship exists between the two endpoints unless the schema lists it.
```

### A.5 Answer Agent

Answers grounded only on the retrieval results (structured rows, relationships, document
passages). Enforces provenance/citation, closed-world negation (answer "No" if absent from the
retrieved list), and correlation ≠ causation. For the isolation study this instruction is shared
**byte-for-byte** with the baselines (HippoRAG/HybridRAG).

```
You are a manufacturing/supply-chain analyst answering from a single knowledge graph
that fuses relational tables and company documents. The retrieval results below are the
output of a retrieval step that already encodes the question's constraints, so when the
results contain rows or records that answer the question, report that answer directly — do
NOT reply that you could not find it while matching rows are present, and do not demand that
the linking reasoning be restated in the results. Use all of the retrieved evidence
(structured rows, relationships, and any document passages); a single well-supported result
from one source is sufficient. Apply the yes/no and closed-world rules below.

Answer rules:
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
- Answer concisely in the same language as the question.
```

---

## B. Baseline Configurations

Fairness principles: HippoRAG and LightRAG run on the **same fused KG as IBKG** (with Entity /
Relation Bound applied) with only the retriever swapped, and all baselines share the **same
generation LLM, the same Answer instruction, and the same grader**. Structured properties are
never withheld from the context.

### B.1 HippoRAG (PPR)

Question NER → node linking (exact ID/name + embedding nearest-neighbor) → **Personalized
PageRank** propagation → passage (document chunk) scoring, top-k.

| Parameter | Value | Description |
|---|---|---|
| `k_passages` | 12 | number of retrieved passages |
| `m_entities` | 15 | top entities included in context |
| `alpha` | 0.15 | PPR teleport (damping) |
| `iters` | 30 | PPR iterations |
| `link_tau` | 0.5 | minimum cosine floor for linking |
| `link_margin` | 0.05 | top1−top2 margin (blocks generic-phrase seeds) |

NER prompt:
```
Extract the named entities and specific noun phrases from the question that should be
looked up in a manufacturing/supply-chain knowledge graph (ids like PRD001/SUP005/DT0007/
PO.../EQ..., product/supplier/equipment names, and concrete concepts).
Return JSON {"entities": ["..."]}. Do not include generic words.
```

### B.2 LightRAG (Dual 1-hop)

Extracts dual-level keywords (low = concrete entities, high = higher-level relation concepts) →
low matched to entity nodes, high matched to relation types via embeddings → collects the 1-hop
relations of matched nodes + edges of matched relation types + the connected chunks.

| Parameter | Value | Description |
|---|---|---|
| `k_low` | 5 | low keywords → entity matches |
| `k_type` | 3 | high keywords → relation-type matches |
| `ent_max` / `r_max` | 30 / 30 | context entity / relation caps |
| `chunk_max` | 8 | collected-chunk cap |
| `link_tau` / `link_margin` | 0.5 / 0.05 | same linking gate as HippoRAG |

Keyword-extraction prompt:
```
Extract keywords from the question at two levels for retrieval over a manufacturing/
supply-chain knowledge graph.
- low_level: concrete entities/specifics (ids like PRD001/SUP005/DT0007, product/supplier/
  equipment names, specific fields).
- high_level: overarching themes / relation concepts (e.g. 'root cause', 'downtime',
  'procurement', 'supplier attribution', 'contract', 'complaint').
Return JSON {"low_level": [...], "high_level": [...]}.
```

### B.3 HybridRAG (KG + Vector)

**Does not use a unified KG.** Separately runs structured-only Text2Cypher (structured edges
only) and document vector search (Chroma top-k), then naively concatenates the two result sets
for synthesis.

| Parameter | Value |
|---|---|
| vector top-k | 6 |
| structured retrieval | `VanillaStructuredRetriever` (structured relations only, Text2Cypher) |
| Answer instruction | `SYNTHESIS_SYSTEM` (identical to IBKG) |

Context layout: `[A. KG results (structured, Text2Cypher)]` + `[B. Document passages
(semantic search)]`.

### B.4 RouteRAG (Question Routing)

Routes each question to one of four paths without fusion (the rule-driven A_ROUTING of Chen et
al., WWW'26). An LLM applies a fixed rule set to score the paths → argmax (with tie-break). The
rule self-improvement loop and query cache are disabled to isolate routing quality.

- Paths: `DB` (structured Text2Cypher) / `Doc` (document vector) / `Hybrid` (both) / `LLM`
  (no retrieval)
- Tie-break priority: `Hybrid > DB > Doc > LLM` (the most-evidence path is placed first, favoring
  the baseline)
- Additive rules (summary): specific record/id/quantity/status → DB+3 / narrative note content,
  why/how → Doc+3 / needs to connect a structured record to a document fact → Hybrid+3 / general
  definition → LLM+3 / mentions both a structured entity and a document concept → Hybrid+2 /
  single-table lookup → DB+1

### B.5 Shared vector store (Doc path of HybridRAG/RouteRAG)

| Item | Value |
|---|---|
| chunking | sentence sliding window `size=3`, `step=1` (documents ≤3 sentences → one whole chunk) |
| embedding | `Qwen3-Embedding-0.6B` (vLLM :8001) |
| index | ChromaDB, HNSW, cosine |
| vector top-k | 6 |

---

## C. Synthetic Dataset

A manufacturing / supply-chain synthetic dataset modeled on Walmart M5. Released together with
the code.

### C.1 Structured data (relational tables)

| Table | Rows | Content |
|---|---:|---|
| `products` | 20 | product master |
| `suppliers` | 40 | suppliers |
| `product_suppliers` | 62 | product–supplier mapping (junction) |
| `customers` | 240 | customers |
| `regions` | 8 | regions |
| `lines` | 3 | production lines (L001–L003) |
| `equipment` | 40 | equipment |
| `contracts` | 180 | contracts |
| `purchase_orders` | 700 | purchase orders |
| `purchase_order_lines` | 811 | purchase-order lines |
| `downtime_events` | 114 | downtime events |
| `anomaly_events` | 212 | anomaly events |
| `service_tickets` | 500 | service tickets |

### C.2 Unstructured data (document corpus)

- `documents`: **196** documents (equipment/supply issues, maintenance logs, ops incident notes,
  procurement notes, CRM cases, contract notes, quality reviews, etc.). Columns: `document_id,
  title, document_date, source, text`.
- `document_links`: **196** links (document ↔ structured-entity evidence links).

### C.3 Question set (171 total)

| Category | Sub-type | Count |
|---|---|---:|
| **Bridge** (structured–unstructured–structured, 3-hop) | bridge-BT1 … BT6 (15 each) | 90 |
| **Structured** (structured 1/2/3-hop) | structured-only 15 · structured-2hop 8 · structured-3hop 7 | 30 |
| **Document** (unstructured facts) | doc-only 15 · mixed-2hop 15 | 30 |
| **Trap** (detect misleading paths; answer usually "No") | trap-negative 15 · polysemy-trap 6 | 21 |
| **Total** | | **171** |

The six bridge types (BT1–BT6) correspond one-to-one to the six bridge Label pairs in A.2. The
polysemy-trap set (6) contains, for the three pairs with polar-opposite senses (BT1/BT3/BT5),
two questions each — traps whose wording is similar but whose direction/causality is reversed.

### C.4 Grading

If the answer at the end of the Gold Path appears as a substring of the final answer text, it is
scored Pass, otherwise Fail → **Accuracy**. Multi-answer cases (`A;B;C`) require all parts to be
present, and the YES/NO of trap questions is scored by polarity.
