"""
retrieve_d.py — 전략 D: 시드 링킹 + 전역 관계 프루닝 -> '유계 스키마'로 Text2Cypher.

흐름:
  1) 시드 링킹    : 질문에서 엔티티 링킹(ID 정확 + 이름 부분매칭).
  2) 프루닝       : 전역 관계를 (SourceLabel)-[:REL]->(TargetLabel) 형태로 LLM에 주고
                    관련 관계만 고르게 함. 이름만 주면 판단이 안 되므로 라벨을 함께 준다.
  3) 유계 스키마  : 선택된 관계 + 거기 등장하는 라벨만으로 스키마를 **재구성**해 프롬프트에 주입.
                    (전역 스키마를 통째로 주면 소형 모델이 관계를 지어내고 문법이 깨진다)
  4) 생성/검증/실행: 읽기전용 + 자동 LIMIT. 문법오류/미존재 관계는 에러를 되먹여 재시도.
  5) provenance   : source_chunk_id -> Chunk 원문 회수.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field

from graph import Neo4jClient, EntityCatalog, CypherSafetyError
from llm import LLMClient
from schema import build_schema_text, relationship_catalog, build_rel_glosses


@dataclass
class Retrieval:
    strategy: str = "D"
    context_text: str = ""
    facts: list = field(default_factory=list)
    cypher: str = ""
    meta: dict = field(default_factory=dict)


_PRUNE_SCHEMA = {
    "type": "object",
    "properties": {"relationship_types": {"type": "array", "items": {"type": "string"}}},
    "required": ["relationship_types"],
}
_PRUNE_SYSTEM = (
    "You are selecting which graph relationships are needed to answer a question. "
    "You are given lines of the form (SourceLabel)-[:REL_TYPE]->(TargetLabel)  [n=<edge count>]. "
    "Return the REL_TYPE names that could lie on a path from the question's entities to the answer. "
    "Include every step of the path, not just the final hop. When unsure, include it. "
    "If several REL_TYPEs look like they express the same idea, prefer the one(s) with the larger n "
    "unless the question's own wording clearly matches a specific lower-n type. "
    "Return JSON {\"relationship_types\": [\"A\",\"B\"]} using only REL_TYPE names from the list."
)

_CYPHER_SCHEMA = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}},
    "required": ["cypher"],
}
_GEN_SYSTEM = """You write ONE read-only Neo4j Cypher query answering the question, using ONLY the schema given.
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
  and never put a relationship type inside a node's {{...}} braces.
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
- Output exactly one JSON object: {{"cypher": "..."}}"""


FEWSHOT = """Examples of correct SYNTAX for this graph. These use different entities/relations                                                                                                                                                                            
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
  direct shortcut relationship exists between the two endpoints unless the schema lists it."""

# 알려진 브리지 라벨쌍(무방향). 이 쌍을 잇는 extracted 타입은 프루닝 recall 바닥으로 항상 스키마에 포함.
# extract.py 의 _ENDPOINT_SENSES/_ENDPOINT_CANON 과 동일한 6쌍(정규화 모드와 무관하게 라벨쌍으로 식별).
_BRIDGE_PAIRS = frozenset({
    frozenset({"DowntimeEvent", "PurchaseOrder"}),
    frozenset({"DowntimeEvent", "Contract"}),
    frozenset({"PurchaseOrder", "AnomalyEvent"}),
    frozenset({"ServiceTicket", "PurchaseOrder"}),
    frozenset({"Contract", "Product"}),
    frozenset({"ServiceTicket", "AnomalyEvent"}),
})

# 방법2(untyped): 아래 쌍은 문서-추출 엣지로만 이어지는데 대표명이 불안정(같은 관계가 여러 이름).
# 이 홉은 이름을 찍지 말고 무방향·무타입으로 매칭하고 origin으로 거른다.
_UNTYPED_HINT = (
    "Some entity pairs are connected ONLY by relationships extracted from documents, whose exact "
    "relationship-type name is unreliable (the same relation is stored under many different names). "
    "For a hop between such a pair, do NOT name the relationship type — match it UNTYPED and filter "
    "by origin:\n"
    "    (A)-[r]-(B) WHERE r.origin = 'extracted' AND r.confidence >= {tau}\n"
    "and RETURN type(r) AS rel and r.source_chunk_id AS source_chunk_id so the meaning can be "
    "verified. All other (structured) hops keep their exact type as usual.\n"
    "Pairs to traverse untyped:\n{pairs}")
_UNTYPED_FEWSHOT = """

  Q (bridge with an untyped document hop in the middle):
     On which equipment was the anomaly that supplier SUP004's expedited purchase order addressed?
  A: MATCH (s:Supplier {{supplier_id: 'SUP004'}})<-[:FROM_SUPPLIER]-(po:PurchaseOrder)
     MATCH (po)-[r]-(ae:AnomalyEvent) WHERE r.origin = 'extracted' AND r.confidence >= {tau}
     MATCH (ae)-[:ON_EQUIPMENT]->(e:Equipment)
     RETURN e.equipment_id AS equipment_id, type(r) AS rel, r.source_chunk_id AS source_chunk_id"""


class SubgraphText2CypherRetriever:
    strategy_name = "D"

    def __init__(self, graph: Neo4jClient, llm: LLMClient, catalog: EntityCatalog,
                 tau: float = 0.5, retries: int = 2, debug: bool = False,
                 min_extracted_support: int = 2, prune_mode: str = "llm",
                 dedup_structured: bool = True, bridge_match: str = "typed"):
        self.graph = graph
        self.llm = llm
        self.catalog = catalog
        self.tau = tau
        self.retries = retries
        self.debug = debug
        self.min_extracted_support = min_extracted_support
        # 프루닝 모드: "llm"=LLM 관련도 프루닝(+브리지 recall 바닥), "none"=프루닝 없이 전체 스키마
        self.prune_mode = prune_mode
        # 브리지 가운데 홉 매칭: "typed"=대표명으로 조회 | "untyped"=이름 무관 -[r]- WHERE origin='extracted'
        # (정규화가 만든 대표명 오류·파편화에 면역 — 방법2)
        self.bridge_match = bridge_match
        self.rel_catalog = relationship_catalog(graph)          # (a, rel, b, origin, count)
        if dedup_structured:                                    # 구조화 쌍 위 extracted 중복(미끼) 제거
            self.rel_catalog = self._dedup_structured_dups(self.rel_catalog)
        self.all_types = sorted({x[1] for x in self.rel_catalog})
        self.rel_gloss = build_rel_glosses(graph, self.rel_catalog)   # 관계 의미 주석(1회 계산·캐시)
        # 정형/문서링크 관계는 그래프의 '뼈대'이고 종류가 적다 -> 프루닝 대상에서 제외하고 항상 포함.
        # (프루닝에 맡기면 소형 모델이 이름이 그럴듯한 extracted 관계를 고르고 정형을 버린다 — 실측)
        self.backbone_types = sorted({x[1] for x in self.rel_catalog
                                      if x[3] in ("structured", "document_link")})
        # extracted 관계타입별 총 지지 엣지 수(해당 타입이 걸치는 모든 (a,b) 패턴의 count 합).
        # min_extracted_support 미만인 타입은 오타/환각성 1회성 predicate일 가능성이 높아
        # 후보에서 제외한다(backbone은 이 필터 대상이 아님 — 정형데이터는 희소해도 사실이다).
        support: dict[str, int] = {}
        for (a, t, b, o, c) in self.rel_catalog:
            if t in set(self.backbone_types):
                continue
            support[t] = support.get(t, 0) + c
        self.type_support = support
        all_extracted = sorted({x[1] for x in self.rel_catalog
                                if x[1] not in set(self.backbone_types)})
        self.extracted_types = [t for t in all_extracted if support.get(t, 0) >= min_extracted_support]
        self.dropped_low_support = [t for t in all_extracted if support.get(t, 0) < min_extracted_support]
        if self.dropped_low_support:
            print(f"  [prune] extracted {len(all_extracted)}종 중 지지도 {min_extracted_support} 미만 "
                  f"{len(self.dropped_low_support)}종 후보 제외 (조용히 버리지 않음 — 아래 목록)")
            if self.debug:
                print(f"    {self.dropped_low_support}")
        # 검증용 인덱스: 유효한 (소스라벨, 관계, 타겟라벨) 조합 / 라벨 / 라벨별 속성
        self.valid_patterns = {(a, t, b) for (a, t, b, o, c) in self.rel_catalog}
        self.valid_labels = {x[0] for x in self.rel_catalog} | {x[2] for x in self.rel_catalog}
        try:
            self.node_props = graph.node_properties()
        except Exception:
            self.node_props = {}
        # untyped 모드용: 그래프에 extracted 엣지가 실제 있는 브리지 라벨쌍만 힌트로.
        present = {frozenset({a, b}) for (a, t, b, o, c) in self.rel_catalog if o == "extracted"}
        hint_pairs = [sorted(fs) for fs in _BRIDGE_PAIRS if fs in present]
        self.bridge_hint = "\n".join(
            f"    ({p[0]}) ~ ({p[1] if len(p) > 1 else p[0]})" for p in sorted(hint_pairs))

    # 1) 시드
    def _seeds(self, question: str) -> list[tuple[str, str]]:
        return self.catalog.find_in_text(question)

    # 2) 프루닝 (라벨 포함 패턴을 보여줘야 제대로 고른다)
    def _prune(self, question: str) -> tuple[list[str], list[str]]:
        """extracted 관계만 프루닝한다. backbone(정형/문서링크)은 항상 포함.
        extracted_types는 이미 min_extracted_support 필터를 통과한 것들이고, 각 후보 줄에
        타입의 총 지지 엣지 수(n=)를 붙여 근접 중복 중 어느 쪽이 더 뒷받침되는지 모델이
        판단할 신호를 준다(수만 보고 자동 선택하지 않음 — 최종 판단은 LLM에게 맡김).
        반환: (최종 사용할 타입 전체, LLM이 고른 extracted 타입)"""
        ext = sorted({(a, t, b) for (a, t, b, o, c) in self.rel_catalog
                      if t in set(self.extracted_types)})
        if not ext:
            return list(self.backbone_types), []
        patterns = sorted(
            {f"({a})-[:{t}]->({b})  [n={self.type_support.get(t, 0)}]" for (a, t, b) in ext},
            key=lambda s: -int(s.rsplit("n=", 1)[1].rstrip("]")),
        )
        # 고정 후보목록을 앞에(질문을 맨 뒤로) 두어 vLLM prefix-cache(APC)가
        # 후보목록 KV를 질문 간 재사용하게 한다. 질문만 매번 새로 prefill.
        # PRUNE_QUESTION_FIRST=1 이면 옛 순서(질문 먼저 → 후보목록 캐시 안 됨) — A/B 비교용.
        if os.getenv("PRUNE_QUESTION_FIRST") in ("1", "true", "True"):
            user = (f"Question: {question}\n\n"
                    f"[Candidate relationships ({len(patterns)})]\n" + "\n".join(patterns))
        else:
            user = (f"[Candidate relationships ({len(patterns)})]\n" + "\n".join(patterns)
                    + f"\n\nQuestion: {question}")
        try:
            out = self.llm.chat_json(_PRUNE_SYSTEM, user, _PRUNE_SCHEMA,
                                     list_key="relationship_types", max_tokens=400)
            allow = set(self.extracted_types)
            sel = [t for t in (out.get("relationship_types") or []) if t in allow]
        except Exception as e:
            print(f"  [warn] 프루닝 실패({type(e).__name__}) -> extracted 전체 사용")
            sel = list(self.extracted_types)
        # recall 바닥: 알려진 브리지 라벨쌍을 잇는 extracted 타입은 LLM이 빠뜨려도 항상 포함.
        floor = self._bridge_floor()
        added = floor - set(sel)
        if added and self.debug:
            print(f"  [prune] 브리지 recall 바닥으로 추가: {sorted(added)}")
        sel = sorted(set(sel) | floor)
        return sorted(set(self.backbone_types) | set(sel)), sel

    @staticmethod
    def _dedup_structured_dups(cat: list) -> list:
        """구조화 관계가 이미 있는 라벨쌍의 extracted 패턴을 제거한다.
        문서가 구조화 사실을 재서술해 생긴 잉여 중복(RECORDED_AS=ON_EQUIPMENT 등)이 미끼가 되어
        생성기를 불완전한 경로로 유인 -> 0행. 구조화가 완전하므로 제거해도 정보 손실 0.
        브리지 6쌍은 전부 비-구조화 라벨쌍이라 보존된다."""
        structured_pairs = {frozenset({a, b}) for (a, t, b, o, c) in cat if o == "structured"}
        kept, dropped = [], []
        for x in cat:
            (a, t, b, o, c) = x
            if o == "extracted" and frozenset({a, b}) in structured_pairs:
                dropped.append(t)
            else:
                kept.append(x)
        if dropped:
            from collections import Counter
            print(f"  [dedup] 구조화 쌍 위 extracted 패턴 {len(dropped)}개 제거(잉여 중복): "
                  f"{dict(Counter(dropped))}")
        return kept

    def _bridge_floor(self) -> set:
        """알려진 브리지 라벨쌍을 잇는 extracted 타입 집합. 정규화 모드와 무관하게
        그래프에 실제 존재하는 타입 이름을 그대로 집어온다(min_support 통과분만)."""
        ext = set(self.extracted_types)
        return {t for (a, t, b, o, c) in self.rel_catalog
                if t in ext and frozenset({a, b}) in _BRIDGE_PAIRS}

    # 3)+4) 생성
    def _gen_cypher(self, question, seeds, schema_text, error) -> str:
        seed_lines = "\n".join(
            f"  ({lbl} {{{_pk(lbl)}: '{sid}'}})" +
            (f"   // {self.catalog.name_of(lbl, sid)}" if self.catalog.name_of(lbl, sid) else "")
            for (lbl, sid) in seeds
        ) or "  (no entity id found in the question)"
        bridge_block = ""
        if self.bridge_match == "untyped" and self.bridge_hint:
            bridge_block = ("[Document-extracted bridges — traverse UNTYPED]\n"
                            + _UNTYPED_HINT.format(tau=self.tau, pairs=self.bridge_hint)
                            + _UNTYPED_FEWSHOT.format(tau=self.tau) + "\n\n")
        user = (f"[Schema — use only these]\n{schema_text}\n\n"
                f"[Few-shot]\n{FEWSHOT}\n\n"
                f"{bridge_block}"
                f"[Seed entities from the question]\n{seed_lines}\n\n"
                f"Reminder: copy every relationship type and node label character-for-character "
                f"from the schema block above. Do not paraphrase, abbreviate, or retype them from "
                f"memory.\n\n"
                f"Question: {question}")
        if error:
            user += (f"\n\nYour previous attempt was invalid. Keep everything that was correct and "
                     f"change ONLY what this message says. Output the full corrected query.\n"
                     f"Problem: {error[:300]}")
        out = self.llm.chat_json(_GEN_SYSTEM.format(tau=self.tau), user,
                                 _CYPHER_SCHEMA, max_tokens=700)
        return _clean_cypher(out.get("cypher", ""))

    def _autofix_tokens(self, cypher: str, allowed, open_ch: str,
                        min_ratio: float = 0.75) -> tuple[str, list[tuple[str, str]]]:
        """[:TOKEN] 또는 (:TOKEN) 형태로 쓰인 토큰이 allowed에 없지만, 편집 유사도가
        min_ratio 이상인 후보가 allowed 안에 정확히 1개뿐이면 그걸로 고쳐 쓴다
        (SUPPLIEDBY -> SUPPLIED_BY, AFFECTCTED_PRODUCT -> AFFECTED_PRODUCT 같은 밑줄 누락/
        문자 중복 오타를 재시도 없이 즉시 복구). 후보가 애매하면(2개 이상) 건드리지 않고
        그대로 둬서 _check_cypher가 명시적 에러로 잡게 한다 — 잘못된 확신으로 고치지 않음."""
        allowed_set = set(allowed)
        pattern = re.compile(rf"({re.escape(open_ch)}\s*\w*\s*:\s*)`?([A-Za-z_][A-Za-z0-9_]*)`?")
        fixes: list[tuple[str, str]] = []

        def repl(m: "re.Match") -> str:
            prefix, tok = m.group(1), m.group(2)
            if tok in allowed_set:
                return m.group(0)
            # n=2로 뽑아서 "커트라인을 넘는 후보가 정확히 1개"인지 확인한다.
            # n=1로 뽑으면 애매한 경우(후보 2개 이상이 커트라인 이상)도 그냥 1등만 반환돼서
            # 조용히 잘못 고칠 위험이 있다 — 그래서 여기서 명시적으로 개수를 센다.
            close = difflib.get_close_matches(tok, allowed_set, n=2, cutoff=min_ratio)
            if len(close) == 1:
                fixes.append((tok, close[0]))
                return f"{prefix}{close[0]}"
            return m.group(0)

        return pattern.sub(repl, cypher), fixes

    def _check_cypher(self, cypher: str, allowed: list[str]) -> str | None:
        """실행 전 정적 검증. 존재하지 않는 관계타입/라벨/속성/끝점조합을 잡아
        '구체적인' 에러를 돌려준다(막연한 파서 에러보다 모델이 잘 고친다)."""
        # 1) 관계 타입
        used_rels = set(re.findall(r"\[\s*\w*\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", cypher))
        bad = sorted(t for t in used_rels if t not in set(self.all_types))
        if bad:
            return (f"relationship type(s) {bad} do not exist. "
                    f"Use only these: {', '.join(allowed)}")
        # 2) 노드 라벨
        used_labels = set(re.findall(r"\(\s*\w*\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", cypher))
        badl = sorted(l for l in used_labels if self.valid_labels and l not in self.valid_labels)
        if badl:
            return (f"node label(s) {badl} do not exist. "
                    f"Use only these: {', '.join(sorted(self.valid_labels))}")
        # 3) 끝점 조합: 절(MATCH/WHERE/RETURN...) 단위로 끊고, 라벨 없는 변수는
        #    앞서 바인딩된 라벨로 해석해서 각 홉을 검사한다.
        var_label = {}
        for v, lb in re.findall(r"\(\s*(\w+)\s*:\s*(\w+)", cypher):
            var_label.setdefault(v, lb)
        node_re = re.compile(r"\(\s*(\w*)\s*(?::\s*(\w+))?\s*(?:\{[^}]*\})?\s*\)")
        rel_re = re.compile(r"(<-)?\s*\[\s*\w*\s*:\s*(\w+)[^\]]*\]\s*(->)?")
        clause_re = re.compile(r"\b(OPTIONAL\s+MATCH|MATCH|WHERE|RETURN|WITH|UNWIND|ORDER\s+BY|LIMIT)\b",
                               re.IGNORECASE)
        for segment in clause_re.split(cypher):
            if not segment or clause_re.fullmatch(segment.strip()):
                continue
            seq = []
            for m in re.finditer(f"({node_re.pattern})|({rel_re.pattern})", segment):
                txt = m.group(0)
                nm = node_re.fullmatch(txt.strip())
                if nm:
                    var, lb = nm.group(1), nm.group(2)
                    seq.append(("node", lb or var_label.get(var)))
                    continue
                rm = rel_re.fullmatch(txt.strip())
                if rm:
                    seq.append(("rel", rm.group(2), bool(rm.group(1)), bool(rm.group(3))))
            for i in range(1, len(seq) - 1):
                if seq[i][0] != "rel" or seq[i - 1][0] != "node" or seq[i + 1][0] != "node":
                    continue
                la, lb2 = seq[i - 1][1], seq[i + 1][1]
                if not la or not lb2:          # 라벨을 알 수 없으면 검증 생략
                    continue
                _, rel, left, right = seq[i]
                if left and not right:
                    src, tgt = lb2, la
                elif right and not left:
                    src, tgt = la, lb2
                else:
                    continue
                if (src, rel, tgt) not in self.valid_patterns:
                    ok = sorted({f"({a})-[:{t}]->({b})" for (a, t, b) in self.valid_patterns if t == rel})
                    return (f"({src})-[:{rel}]->({tgt}) does not exist in the graph. "
                            f"{rel} only connects: {'; '.join(ok) if ok else '(nothing)'}")
        # 4) 속성명: var:Label 로 바인딩된 변수의 var.prop 검사
        if self.node_props:
            var_label = dict(re.findall(r"\(\s*(\w+)\s*:\s*(\w+)", cypher))
            # (a) 노드 패턴 안의 인라인 속성:  (p:Product {list_id: 'X'})
            for lb, body in re.findall(r"\(\s*\w*\s*:\s*(\w+)\s*\{([^}]*)\}", cypher):
                if lb not in self.node_props:
                    continue
                for key in re.findall(r"(\w+)\s*:", body):
                    if key not in self.node_props[lb]:
                        return (f"property '{key}' does not exist on ({lb}). "
                                f"Available: {', '.join(sorted(self.node_props[lb]))}")
            # (b) variable.property 접근
            for var, prop in re.findall(r"\b(\w+)\.(\w+)", cypher):
                lb = var_label.get(var)
                if not lb or lb not in self.node_props:
                    continue
                if prop not in self.node_props[lb]:
                    return (f"property '{prop}' does not exist on ({lb}). "
                            f"Available: {', '.join(sorted(self.node_props[lb]))}")
        # 5) 노드 패턴 안의 중복 키
        for body in re.findall(r"\{([^}]*)\}", cypher):
            keys = re.findall(r"(\w+)\s*:", body)
            if len(keys) != len(set(keys)):
                return "a node pattern repeats the same property key; use one key per node pattern"
        return None

    # 5) provenance
    def _provenance(self, rows: list[dict]) -> tuple[str, list[str]]:
        ids = set()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, str) and "#c" in v and "chunk" in k.lower():
                    ids.add(v)
                elif isinstance(v, list):
                    ids.update(x for x in v if isinstance(x, str) and "#c" in x)
        if not ids:
            return "", []
        chunks = self.graph.fetch_chunks(sorted(ids))
        return ("\n".join(f"  [{c['chunk_id']}] {c.get('text','')}" for c in chunks),
                sorted(ids))

    def retrieve(self, question: str, force_rel_types: list[str] | None = None,
                 force_cypher: str | None = None) -> Retrieval:
        seeds = self._seeds(question)
        if force_rel_types is not None:          # ablation: 프루닝 우회, 스키마 스코프 강제 주입
            pruned = sorted(set(force_rel_types))
            sel_ext = [t for t in pruned if t not in set(self.backbone_types)]
        elif self.prune_mode == "none":          # 프루닝 없이 전체 스키마(그래프 전 관계타입) 사용
            pruned = sorted(self.all_types)
            sel_ext = [t for t in pruned if t not in set(self.backbone_types)]
        else:
            pruned, sel_ext = self._prune(question)
        seed_labels = [lbl for (lbl, _) in seeds]
        schema_text = build_schema_text(self.graph, tau=self.tau, rel_types=pruned,
                                        catalog=self.rel_catalog, extra_labels=seed_labels,
                                        glosses=self.rel_gloss)
        if self.debug:
            print(f"  [seeds] {seeds}")
            print(f"  [scope] backbone {len(self.backbone_types)} + extracted 선택 {len(sel_ext)}"
                  f" = {len(pruned)}/{len(self.all_types)}")
            print(f"  [extracted 선택] {sel_ext}")

        error, last_cypher, tried = None, "", []
        for attempt in range(self.retries + 1):
            if force_cypher is not None:          # ablation: 생성 건너뛰고 gold Cypher 직접 실행
                cypher = _clean_cypher(force_cypher)
            else:
                cypher = self._gen_cypher(question, seeds, schema_text, error)
            if cypher:
                cypher, rel_fixes = self._autofix_tokens(cypher, pruned, "[")
                cypher, label_fixes = self._autofix_tokens(cypher, self.valid_labels, "(")
                if rel_fixes or label_fixes:
                    print(f"  [autofix] relation {rel_fixes} label {label_fixes}")
            last_cypher = cypher
            tried.append(cypher)
            if self.debug:
                print(f"  [attempt {attempt+1}] {cypher[:200]}")
            if not cypher:
                error = "empty query"
                continue
            bad = self._check_cypher(cypher, pruned)
            if bad:
                error = bad
                if force_cypher is not None:      # 고정 쿼리는 재시도 무의미 -> 즉시 종료
                    break
                continue
            try:
                safe = self.graph.validate(cypher)
                rows = self.graph.run_read(safe, validate=False)
                context = _serialize(rows)
                prov, prov_ids = self._provenance(rows)
                if prov:
                    context += "\n\n[source text for extracted relationships]\n" + prov
                return Retrieval(
                    context_text=context, facts=rows, cypher=safe,
                    meta={"seeds": seeds, "pruned_types": pruned,
                          "selected_extracted": sel_ext,
                          "n_backbone": len(self.backbone_types),
                          "n_all_types": len(self.all_types), "attempts": attempt + 1,
                          "provenance_chunks": prov_ids, "tried": tried,
                          "schema_text": schema_text if self.debug else None},
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if self.debug:
                    print(f"  [error] {error[:200]}")
                if force_cypher is not None:
                    break

        return Retrieval(
            context_text="(failed to generate a valid Cypher query)", cypher=last_cypher,
            meta={"seeds": seeds, "pruned_types": pruned, "selected_extracted": sel_ext,
                  "n_backbone": len(self.backbone_types), "n_all_types": len(self.all_types),
                  "error": error, "attempts": self.retries + 1, "tried": tried},
        )


def _pk(label: str) -> str:
    from graph import LABEL_PK
    return LABEL_PK.get(label, "id")


def _clean_cypher(cypher: str) -> str:
    c = (cypher or "").strip()
    c = re.sub(r"^```(?:cypher)?\s*", "", c)
    c = re.sub(r"\s*```$", "", c)
    return _undirect(c.strip())


def _undirect(cypher: str) -> str:
    """관계 방향(화살표)을 제거해 방향 역전 오류(원인 C)를 실행 단계에서 무력화한다.

    retrieval은 읽기 전용이고 이 스키마의 각 predicate는 방향이 유일하므로
    (예: SUPPLIED_BY는 항상 Product->Supplier) 무방향 매칭이 결과를 바꾸지 않는다.
    부작용으로 'relationship (X)-[:R]->(Y) does not exist' 에러가 사라져
    재시도가 방향이 아닌 진짜 원인(파편화/프루닝)에 쓰이게 된다."""
    c = cypher
    c = re.sub(r"-(\[[^\]]*\])->", r"-\1-", c)   # 오른쪽 방향:  -[:R]-> -> -[:R]-
    c = re.sub(r"<-(\[[^\]]*\])-", r"-\1-", c)   # 왼쪽 방향:   <-[:R]-  -> -[:R]-
    c = c.replace("-->", "--").replace("<--", "--")  # 타입 없는 화살표
    return c


def _serialize(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    return json.dumps(rows, ensure_ascii=False, indent=2, default=str)