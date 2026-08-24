"""
extract.py — chunk -> LLM 트리플 추출 -> 해소/정규화 -> confidence -> 승격/쓰기
==============================================================================
순서(확정): chunk -> 추출 -> (검색은 이후 단계).

파이프라인:
  1) 전체 문서를 chunk.py 로 1~3문장 슬라이딩 chunk화.
  2) 각 chunk에서 '후보 엔티티'를 카탈로그로 링킹(ID 정확/이름 부분). 후보 2개 미만이면 skip.
  3) LLM에게 "후보들 사이에, 본문에 명시된 관계만" guided_json으로 추출 요청.
  4) 각 트리플의 양끝을 정형 노드로 해소(실패=버림), predicate를 UPPER_SNAKE로 정규화.
  5) confidence.py 로 스코어링(엔드포인트/원문근거/중복). tau 이상만 '승격'.
  6) 승격 엣지를 origin='extracted'(+confidence, source_doc, source_chunk_id, evidence)로 write.

쓰기 분리: 검색 경로(graph.py)는 읽기전용 유지. 여기서만 쓰기 드라이버 사용.
정규화 훅: predicate 동의어 병합은 _canon_predicate 한 곳에서. 지금은 UPPER_SNAKE + 최소 정리.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from chunk import chunk_corpus, Chunk
import confidence as C


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class ExtractConfig:
    window_size: int = 3
    window_step: int = 1
    min_candidates: int = 2          # chunk 내 엔티티가 이 수 미만이면 추출 skip
    # predicate 파편화 정규화(추출 후 배치패스). off | endpoint_override | embed_cluster | guarded_override
    normalize_mode: str = "off"
    emb_tau: float = 0.82            # embed_cluster: 코사인 이 값 이상이면 같은 관계로 병합
    sense_tau: float = 0.55          # guarded_override: sense 프로토타입과 코사인 이 값 미만이면 매핑 거부(원본 유지)
    sense_margin: float = 0.0        # guarded_override: top1-top2 이 값 미만이면 애매로 보고 거부(0=off)
    # --- Entity 구속 축 (ablation) ---
    #  bound: 백본에 해소되는 엔티티만 유지(=현재, 구속). open: 비-백본 추출 엔티티도 :ExtractedEntity 노드로 유지.
    entities: str = "bound"          # bound | open
    entity_merge: str = "string"     # open 모드 비-백본 엔티티 병합: string(정준 키·결정론) | embed(임베딩 τ)
    entity_tau: float = 0.86         # entity_merge=embed: 코사인 이 값 이상이면 같은 엔티티로 병합
    conf: C.ConfidenceConfig = field(default_factory=C.ConfidenceConfig)


# LLM에 넘길 JSON 스키마(guided_json)
TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {"triples": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string"},
            "predicate": {"type": "string"},
            "object_id": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["subject_id", "predicate", "object_id", "evidence"],
    }}},
    "required": ["triples"],
}

_SYSTEM = (
    "You extract relationships that are EXPLICITLY stated in the given text, only between the "
    "listed candidate entities. Use the exact candidate IDs for subject_id and object_id. "
    "'predicate' is a short verb phrase describing the relation. 'evidence' is the exact span "
    "from the text that states it. Do NOT invent relationships not present in the text. "
    "If there is no explicit relationship, return an empty list. Output JSON only."
)

# open 모드(Entity 축 B): 후보에 없는 엔티티도 허용. 후보에 있으면 그 ID를, 없으면 텍스트 표면구를 그대로 id로.
_SYSTEM_OPEN = (
    "You extract relationships EXPLICITLY stated in the given text. Prefer the listed candidate "
    "entities and use their exact IDs when a related entity is among them. If a related entity is "
    "NOT among the candidates, use its exact surface phrase from the text as the id (verbatim). "
    "'predicate' is a short verb phrase; 'evidence' is the exact span stating it. Do NOT invent "
    "relationships. If none are explicit, return an empty list. Output JSON only."
)


def _entity_key(surface: str) -> str:
    """②정준 문자열 키(최소 정규화): 소문자 · 비영숫자→공백 · 공백 정리. 결정론적 dedup용."""
    s = re.sub(r"[^0-9a-z]+", " ", str(surface).lower())
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 엔티티 링킹(자기완결 버전; 프로덕션에선 graph.EntityCatalog 로 위임 가능)
# catalog: list of (label, id, name)  ->  find: [(label, id, name, match)]
# ---------------------------------------------------------------------------
class Catalog:
    EXT_LABEL = "ExtractedEntity"
    EXT_PK = "ext_id"

    def __init__(self, rows: list[tuple[str, str, str | None]], label_pk: dict[str, str]):
        self.rows = rows
        # open 모드에서 생성되는 :ExtractedEntity 노드의 pk 등록(백본과 동일 파이프라인으로 흐르게)
        self.label_pk = {**label_pk, self.EXT_LABEL: self.EXT_PK}
        self._by_id = {id_: (label, name) for (label, id_, name) in rows}
        self._extracted: dict[str, str] = {}       # synth_id -> 대표 surface (open 모드 신규 엔티티)
        # 이름 -> id (소문자). 모델이 ID 대신 이름을 돌려주는 경우가 많아 필수.
        self._by_name = {}
        for (label, id_, name) in rows:
            if name:
                self._by_name.setdefault(str(name).strip().lower(), id_)

    def find_in_text(self, text: str) -> list[tuple[str, str, str | None, str]]:
        low = (text or "").lower()
        hits: dict[str, tuple[str, str, str | None, str]] = {}
        for (label, id_, name) in self.rows:
            if re.search(rf"\b{re.escape(id_)}\b", text, re.IGNORECASE):
                hits[id_] = (label, id_, name, "id")            # ID 정확매칭 우선
            elif name and name.lower() in low and id_ not in hits:
                hits[id_] = (label, id_, name, "name")          # 이름 부분매칭
        return list(hits.values())

    def resolve(self, id_: str, text: str) -> tuple[str | None, str, str, str | None]:
        """반환: (label, match, surface, name). 실패면 label=None."""
        if id_ in self._by_id:
            label, name = self._by_id[id_]
            if re.search(rf"\b{re.escape(id_)}\b", text, re.IGNORECASE):
                return label, "id", id_, name
            if name and name.lower() in (text or "").lower():
                return label, "name", name, name
            return label, "id", id_, name     # 카탈로그엔 있으나 표면형 확인 실패 -> 근거점수에서 감점됨
        # ID가 아니면 '이름'으로 들어온 경우를 해소 (예: "Harvest Consumer Goods 31" -> SUP031)
        key = str(id_).strip().lower()
        real = self._by_name.get(key)
        if real is None:                       # 부분일치 폴백
            for nm, rid in self._by_name.items():
                if nm and (nm in key or key in nm) and len(key) >= 4:
                    real = rid; break
        if real is not None:
            label, name = self._by_id[real]
            return label, "name", (name or real), name
        return None, "none", id_, None

    def canonical_id(self, value: str) -> str | None:
        """ID면 그대로, 이름이면 대응 ID로. 해소 불가면 None."""
        if value in self._by_id:
            return value
        key = str(value).strip().lower()
        if key in self._by_name:
            return self._by_name[key]
        for nm, rid in self._by_name.items():
            if nm and (nm in key or key in nm) and len(key) >= 4:
                return rid
        return None

    def register_extracted(self, surface: str) -> str | None:
        """②최소 정규화: 비-백본 추출 엔티티를 정준 문자열 키로 dedup 후 :ExtractedEntity 노드로 등록.
        같은 키(소문자·비영숫자→공백·공백정리)면 같은 synth_id 로 병합. 결정론·고정밀."""
        key = _entity_key(surface)
        if not key or len(key) < 2:
            return None
        sid = "X_" + re.sub(r"\s+", "_", key)[:48]
        if sid not in self._by_id:
            disp = str(surface).strip()[:80]
            self._by_id[sid] = (self.EXT_LABEL, disp)
            self._extracted[sid] = disp
        return sid

    @classmethod
    def from_csv(cls, data_dir: str) -> "Catalog":
        """CSV 폴더에서 카탈로그 구성(추출은 빌드타임이라 CSV가 가장 직접적). 있는 파일만."""
        import os
        import pandas as pd
        spec = [  # (file, label, id_col, name_col)
            ("products.csv",        "Product",       "product_id",        "product_name"),
            ("suppliers.csv",       "Supplier",      "supplier_id",       "supplier_name"),
            ("equipment.csv",       "Equipment",     "equipment_id",      "equipment_name"),
            ("lines.csv",           "Line",          "line_id",           "line_name"),
            ("regions.csv",         "Region",        "region_id",         "region_name"),
            ("customers.csv",       "Customer",      "customer_id",       "customer_name"),
            ("anomaly_events.csv",  "AnomalyEvent",  "anomaly_id",        None),
            ("downtime_events.csv", "DowntimeEvent", "downtime_id",       None),
            ("service_tickets.csv", "ServiceTicket", "ticket_id",         None),
            ("purchase_orders.csv", "PurchaseOrder", "purchase_order_id", None),
            ("contracts.csv",       "Contract",      "contract_id",       None),
        ]
        rows, label_pk = [], {}
        for fn, label, idc, namec in spec:
            p = os.path.join(data_dir, fn)
            if not os.path.isfile(p):
                continue
            df = pd.read_csv(p, encoding="utf-8-sig")
            label_pk[label] = idc
            for _, r in df.iterrows():
                name = str(r[namec]) if (namec and namec in df.columns and pd.notna(r.get(namec))) else None
                rows.append((label, str(r[idc]), name))
        return cls(rows, label_pk)


# 명백한 동의어만 정준형으로 병합(표기만 다른 같은 관계). 여기 없는 predicate는 그대로 통과(열린 어휘).
# 다양성은 보존 — 뜻이 다른 관계(ATTRIBUTED_TO vs PART_SUPPLIED_BY vs DELAYED 등)는 안 건드림.
#
# 병합 기준: 방향(주어/목적어 역할)이 같다고 확인된 것만 묶는다. 능동/수동으로 방향이
# 뒤집힐 수 있는 쌍(예: COVERS vs COVERED_BY)은 실데이터 검증 없이 합치지 않는다 —
# 잘못 합치면 관계 방향이 조용히 틀어져서 나중에 발견하기 훨씬 어렵다.
# 오타/붕괴성 토큰(RECORD020, HAS_SERVICEHTAG 류)은 여기서 다루지 않는다 — 대개 지지 엣지
# 수가 1~2건뿐이라 retrieve_d.py의 min_extracted_support 필터가 검색 후보에서 걸러낸다.
_SYNONYMS = {
    "LED_TO":      ["LED_TO", "RESULTED_IN", "TRIGGERED", "CAUSED"],
    "DELAYED":     ["DELAYED", "HELD_UP", "POSTPONED"],
    "SUPPLIED_BY": ["SUPPLIED_BY", "PROVIDED_BY", "SOURCED_FROM"],
    "RECOMMENDS":  ["RECOMMENDS", "RECOMMENDED", "ADVISES", "SUGGESTS"],
    "AFFECTED":    ["AFFECTED", "IMPACTED"],
    # 2026-07-23 실측 그래프(56개 관계타입)에서 방향이 같음을 확인하고 추가.
    "ATTRIBUTED_TO": ["ATTRIBUTED_TO", "ATTRIBUTED_ROOT_CAUSE_TO", "TRACED_TO", "ROOT_CAUSED_BY"],
    "RECORDED_ON":   ["RECORDED_ON", "RECORD_ON", "RECORDLED_ON", "IS_RECORD_OF", "ISRECORD_OF"],
    "RELATED_TO":    ["RELATED_TO", "ISRELATEDTO", "LINKED_TO"],
}
_SYNONYM_MAP = {v: canon for canon, variants in _SYNONYMS.items() for v in variants}


# 방법 1(엔드포인트쌍 override): 아는 브리지 라벨쌍 -> 정준 predicate.
# 양끝 라벨쌍이 아래에 해당하면 LLM이 verb를 뭐로 뽑았든 이 이름으로 덮어쓴다(방향 무관, frozenset 키).
# 안전장치(앵커/극성 게이트)는 현재 의도적으로 비워둠 — 각 쌍은 단일 관계라는 설계 전제에 의존.
# eval_questions_bridge100 의 6개 브리지 타입(BT1~6)을 전부 커버한다.
_ENDPOINT_CANON = {
    frozenset({"DowntimeEvent", "PurchaseOrder"}): "RAISED_REPLACEMENT_ORDER",  # BT1
    frozenset({"DowntimeEvent", "Contract"}):      "MISSED_COMMITMENT_UNDER",    # BT2
    frozenset({"PurchaseOrder", "AnomalyEvent"}):  "EXPEDITED_FOR_ANOMALY",      # BT3
    frozenset({"ServiceTicket", "PurchaseOrder"}): "REPLACEMENT_SOURCED_UNDER",  # BT4
    frozenset({"Contract", "Product"}):            "COMMITS_SKU",                # BT5
    frozenset({"ServiceTicket", "AnomalyEvent"}):  "TRACED_TO_ANOMALY",          # BT6
}

# 방법 1.5(guarded_override, 멀티 sense 라우팅): pair 마다 '정준 predicate + 프로토타입 문장'을 여러 개.
# 추출된 predicate를 임베딩 코사인으로 argmax sense 에 매핑 — 임계(sense_tau) 미달이면 매핑 거부(원본 유지).
# 단일 정준(방법1)의 polysemy 사각지대를 sense 를 K개 두어 해소: 브리지 sense + 알려진 반대의미 sense.
# 프로토타입은 자연어 서술(정준명 자체가 아니라 '의미'를 임베딩에 싣기 위함).
_ENDPOINT_SENSES = {
    # BT1 브리지 vs PS-B 트랩 — 인과 방향이 정반대(다운타임→발주 / 발주지연→다운타임)
    frozenset({"DowntimeEvent", "PurchaseOrder"}): [
        ("TRIGGERED_REPLACEMENT_ORDER",
         "an emergency, substitute, replacement or corrective part was ordered, sourced, procured or "
         "expedited under the purchase order to recover from the downtime (the order supplies the fix)"),
        ("LATE_DELIVERY_CAUSED_DOWNTIME",
         "the purchase order's OWN late, missed or defective delivery starved the line and was the "
         "cause of the downtime (the order itself is the cause, not a fix)"),
    ],
    # BT2 단일 — 다운타임이 계약의 약정 납품을 지연/불이행시킴
    frozenset({"DowntimeEvent", "Contract"}): [
        ("DELAYED_CONTRACT_DELIVERY",
         "the downtime delayed, breached or caused a miss of the delivery volume committed under the contract"),
    ],
    # BT3 브리지 vs PS-A 트랩 — 발주가 이상을 고치려 조달 / 불량납품이 이상을 유발
    frozenset({"PurchaseOrder", "AnomalyEvent"}): [
        ("ORDERED_TO_REMEDY_ANOMALY",
         "the purchase order was placed or expedited AFTER the anomaly, to correct, remedy or source the "
         "replacement part that the anomaly required — the order is the fix, the anomaly the problem"),
        ("DELIVERY_CAUSED_ANOMALY",
         "a defective or wrong delivery received under the purchase order was itself the cause of the "
         "anomaly — the order is the cause, the anomaly the effect"),
    ],
    # BT4 단일 — 티켓의 교체 유닛이 해당 PO로 조달됨
    frozenset({"ServiceTicket", "PurchaseOrder"}): [
        ("REPLACEMENT_SOURCED_UNDER_ORDER",
         "the replacement or substitute unit that resolved or closed the service ticket was sourced, "
         "ordered or procured under the purchase order"),
    ],
    # BT5 브리지 vs PS-C 트랩 — 계약이 제품 물량을 약정 / 제품이 계약에서 제외(구분어: include vs exclude)
    frozenset({"Contract", "Product"}): [
        ("COMMITS_PRODUCT_VOLUME",
         "the contract commits, guarantees, locks in or covers a fixed annual volume of this product; "
         "the product IS included in the agreement's committed volume"),
        ("EXCLUDES_PRODUCT",
         "this product is explicitly carved out and EXCLUDED from the contract's committed volume; "
         "it is NOT covered by the commitment"),
    ],
    # BT6 단일 — 티켓이 엔지니어링으로 에스컬레이션되어 이상으로 근본원인 규명
    frozenset({"ServiceTicket", "AnomalyEvent"}): [
        ("ROOT_CAUSED_TO_ANOMALY",
         "the service ticket or complaint was escalated to engineering and its root cause was traced "
         "or tied to the anomaly"),
    ],
}

# 정준명 -> 프로토타입(의미 서술) 평탄화 맵. Text2Cypher 스키마의 gloss로 재사용된다(schema.build_rel_glosses).
# evidence 자동추출과 달리 오프라인 수기라 깨끗하고 sense-split 오염이 없다.
SENSE_GLOSS = {name: proto for senses in _ENDPOINT_SENSES.values() for (name, proto) in senses}


# ---------------------------------------------------------------------------
# 정형 스키마 = FK 관계의 '자동 인벤토리'(동적 FK-fold).
# 정형 backbone이 label쌍마다 대표 관계를 이미 규정하므로, 그 쌍에 온 extracted predicate는
# 수기 없이 이 인벤토리에 대고 판단해 정형 관계명으로 접는다. 데이터셋 교체 시 build 스펙만 고치면 일반화.
# 단일 소스 = build_graph_neo4j.REL_SPECS/JUNCTION_SPECS, gloss = schema._STRUCTURED_GLOSS.
# ---------------------------------------------------------------------------
_STRUCT_PAIR_CACHE: "dict | None" = None


def structured_pair_senses() -> dict:
    """frozenset({A,B}) -> [(REL_NAME, gloss)] : 정형 백본이 규정하는 FK 관계 인벤토리."""
    global _STRUCT_PAIR_CACHE
    if _STRUCT_PAIR_CACHE is not None:
        return _STRUCT_PAIR_CACHE
    m: dict = {}
    try:
        import build_graph_neo4j as _B                          # 지연 임포트(모듈 로드 커플링 회피)
        from schema import _STRUCTURED_GLOSS
        specs = [(s[1], s[4], s[6]) for s in _B.REL_SPECS]      # (src_label, tgt_label, rel_type)
        specs += [(s[1], s[3], s[5]) for s in _B.JUNCTION_SPECS]  # 정션: (src, tgt, rel)
        for a, b, rt in specs:
            if not a or not b or a == b:
                continue
            lst = m.setdefault(frozenset({a, b}), [])
            if rt not in {n for (n, _) in lst}:
                lst.append((rt, _STRUCTURED_GLOSS.get(rt, "")))
    except Exception as e:
        print(f"  [warn] 정형 pair 인벤토리 로드 실패({type(e).__name__}) -> FK-fold 비활성")
    _STRUCT_PAIR_CACHE = m
    return m


_STRUCT_DIR_CACHE: "dict | None" = None


def structured_rel_dir() -> dict:
    """(frozenset({src,tgt}), REL_NAME) -> (src_label, tgt_label) : 정형 관계의 정준 방향.
    같은 관계명이 여러 쌍에 쓰이므로(예: ON_EQUIPMENT = Anomaly/Downtime→Equipment 둘 다) 반드시
    pair로 구분한다. FK-fold로 정형명을 뒤집어쓴 역방향 extracted 엣지를 정형 방향으로 재정렬한다."""
    global _STRUCT_DIR_CACHE
    if _STRUCT_DIR_CACHE is not None:
        return _STRUCT_DIR_CACHE
    d: dict = {}
    try:
        import build_graph_neo4j as _B
        for s in _B.REL_SPECS:
            d.setdefault((frozenset({s[1], s[4]}), s[6]), (s[1], s[4]))   # (pair, rel) -> (src, tgt)
        for s in _B.JUNCTION_SPECS:
            d.setdefault((frozenset({s[1], s[3]}), s[5]), (s[1], s[3]))
    except Exception as e:
        print(f"  [warn] 정형 방향 맵 로드 실패({type(e).__name__}) -> 방향 재정렬 skip")
    _STRUCT_DIR_CACHE = d
    return d


def _canon_predicate(raw: str) -> str:
    """predicate 정규화(열린 어휘 유지). UPPER_SNAKE 표준화 후, 명백한 동의어만 정준형으로 병합.
    _SYNONYM_MAP 에 없는 관계는 그대로 통과 -> 문장에서 뽑히는 대로 다양성 보존."""
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(raw).strip()).strip("_").upper()
    if not t:
        return "RELATED_TO"
    return _SYNONYM_MAP.get(t, t)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
@dataclass
class WriteOp:
    src_label: str; src_id: str; src_pk: str
    rel_type: str
    tgt_label: str; tgt_id: str; tgt_pk: str
    props: dict
    src_name: str | None = None      # open 모드 :ExtractedEntity 노드 표시명(백본이면 None)
    tgt_name: str | None = None


# ---------------------------------------------------------------------------
# 선택 깔때기 계측(loss accounting) — 순수 로깅. 그래프/결과를 바꾸지 않는다.
# bounded fusion에서 위험은 '충돌'이 아니라 '소실'이므로, 각 드롭 게이트에서
# 몇 개가 빠지고 어떤 surface가 백본에 해소 못 됐는지를 기록해 리뷰 방어용 수치로 만든다.
# ---------------------------------------------------------------------------
@dataclass
class FunnelStats:
    raw_llm: int = 0                 # LLM이 뱉은 원 트리플 수
    drop_empty_selfloop: int = 0     # :284 빈값/자기참조
    drop_entity_unresolved: int = 0  # :288 한쪽이라도 백본 해소 실패(=소실)
    drop_canonical_none: int = 0     # :293 정준 ID 치환 실패/자기참조
    resolved: int = 0                # 게이트 통과(=breakdowns 대상)
    unresolved_surfaces: Counter = field(default_factory=Counter)   # 소실된 surface 빈도
    unresolved_records: list = field(default_factory=list)          # 소실 트리플 원본(격리 보관, 다른 데이터셋 대비)
    predicates_before: int = 0       # 정규화 전 distinct predicate 종수
    predicates_after: int = 0        # 정규화 후 distinct predicate 종수
    predicate_counts: Counter = field(default_factory=Counter)      # 정규화 후 predicate 분포
    pair_residual: dict = field(default_factory=dict)               # label쌍 -> 인벤토리 밖(raw 잔존) predicate 목록
    inventory_names: set = field(default_factory=set)               # 수기 sense ∪ 정형 관계명
    genuine_bridge_edges: int = 0    # 정형쌍 없는 엣지 = 테이블로 불가한 연결(=가치)
    genuine_bridge_types: int = 0
    fk_redundant_edges: int = 0      # 정형쌍 있는 엣지(FK-fold로 정형명 흡수)

    def to_dict(self, top: int = 30) -> dict:
        inv = self.inventory_names or set(SENSE_GLOSS)   # 수기 sense ∪ 정형명(없으면 폴백)
        pc = self.predicate_counts
        total = sum(pc.values())
        residual = {p: c for p, c in pc.items() if p not in inv}    # 인벤토리 밖 = 파편 후보
        return {
            "raw_llm_triples": self.raw_llm,
            "drop_empty_or_selfloop": self.drop_empty_selfloop,
            "drop_entity_unresolved": self.drop_entity_unresolved,
            "drop_canonical_none": self.drop_canonical_none,
            "resolved_triples": self.resolved,
            "entity_unresolved_rate": round(self.drop_entity_unresolved / self.raw_llm, 4) if self.raw_llm else 0.0,
            "unresolved_captured": len(self.unresolved_records),
            # --- extracted 엣지 분할: genuine-bridge(정형 불가) vs FK-중복 ---
            "genuine_bridge_edges": self.genuine_bridge_edges,
            "genuine_bridge_types": self.genuine_bridge_types,
            "fk_redundant_edges": self.fk_redundant_edges,
            # --- predicate 파편화 감사 ---
            "predicates_before_norm": self.predicates_before,
            "predicates_after_norm": self.predicates_after,
            "predicate_distinct_total": len(pc),
            "predicate_singletons": sum(1 for c in pc.values() if c == 1),
            "inventory_distinct": len(pc) - len(residual),
            "residual_distinct": len(residual),
            "residual_triples": sum(residual.values()),
            "residual_rate": round(sum(residual.values()) / total, 4) if total else 0.0,
            "residual_top": Counter(residual).most_common(top),
            "pair_residual": self.pair_residual,
            "unresolved_surface_top": self.unresolved_surfaces.most_common(top),
        }


def _extract_chunk(chunk: Chunk, cat: Catalog, llm, cfg: ExtractConfig,
                   stats: "FunnelStats | None" = None) -> list[C.ExtractedTriple]:
    cands = cat.find_in_text(chunk.text)
    if len(cands) < cfg.min_candidates:
        return []
    _open = getattr(cfg, "entities", "bound") == "open"
    cand_lines = "\n".join(f"  {id_} = {name or ''} ({label})" for (label, id_, name, m) in cands)
    user = f"[Candidate entities]\n{cand_lines}\n\n[Text]\n{chunk.text}"
    try:
        out = llm.chat_json(_SYSTEM_OPEN if _open else _SYSTEM, user, TRIPLE_SCHEMA)
    except Exception:
        return []
    triples: list[C.ExtractedTriple] = []
    for tr in (out.get("triples") or []):
        if stats is not None:
            stats.raw_llm += 1
        s_id, o_id = str(tr.get("subject_id", "")), str(tr.get("object_id", ""))
        if not s_id or not o_id or s_id == o_id:
            if stats is not None:
                stats.drop_empty_selfloop += 1
            continue
        s_label, s_match, s_surf, _ = cat.resolve(s_id, chunk.text)
        o_label, o_match, o_surf, _ = cat.resolve(o_id, chunk.text)
        # open 모드(Entity 축 B): ①백본 해소 실패 시 drop 대신 ②정준키로 :ExtractedEntity 유지
        if _open:
            if s_label is None:
                syn = cat.register_extracted(s_id)
                if syn:
                    s_label, s_match, s_surf, s_id = cat.EXT_LABEL, "extracted", str(s_id)[:80], syn
            if o_label is None:
                syn = cat.register_extracted(o_id)
                if syn:
                    o_label, o_match, o_surf, o_id = cat.EXT_LABEL, "extracted", str(o_id)[:80], syn
        if s_label is None or o_label is None:     # E3: 그래프엔 안 넣되, 격리 보관(다른 데이터셋 대비)
            if stats is not None:                  # 어느 쪽이 백본에 없었는지 surface 기록(=소실 증거)
                stats.drop_entity_unresolved += 1
                if s_label is None:
                    stats.unresolved_surfaces[s_id.strip()[:60]] += 1
                if o_label is None:
                    stats.unresolved_surfaces[o_id.strip()[:60]] += 1
                stats.unresolved_records.append({   # 소실 트리플 원본 챙김
                    "doc": chunk.doc_id, "chunk": chunk.chunk_id,
                    "subject": s_id, "object": o_id,
                    "predicate": str(tr.get("predicate", "")),
                    "evidence": str(tr.get("evidence", ""))[:200],
                    "missing": [side for side, miss in
                                (("subject", s_label is None), ("object", o_label is None)) if miss],
                })
            continue
        # 이름으로 들어온 경우 정준 ID로 치환(그래프 키는 항상 ID)
        s_id = cat.canonical_id(s_id)
        o_id = cat.canonical_id(o_id)
        if s_id is None or o_id is None or s_id == o_id:
            if stats is not None:
                stats.drop_canonical_none += 1
            continue
        if stats is not None:
            stats.resolved += 1
        triples.append(C.ExtractedTriple(
            subject_id=s_id, predicate=_canon_predicate(tr.get("predicate")), object_id=o_id,
            subject_match=s_match, object_match=o_match,
            subject_surface=s_surf, object_surface=o_surf,
            predicate_cue=str(tr.get("evidence", "")),
            source_chunk_id=chunk.chunk_id, source_doc_id=chunk.doc_id, chunk_text=chunk.text,
        ))
    return triples


# ---------------------------------------------------------------------------
# predicate 정규화(추출 후 배치패스) — 파편화 완화
# 스코어링 '전'에 도는 이유: 변형을 먼저 합쳐야 corroboration 지지가 한 이름에 모여 confidence가 오른다.
# 라벨쌍(무방향) 스코프로만 병합 — 방향차는 검색단 _undirect가 흡수한다.
# ---------------------------------------------------------------------------
def _labels(cat: Catalog, t: C.ExtractedTriple) -> tuple[str | None, str | None]:
    sl = cat._by_id.get(t.subject_id, (None,))[0]
    ol = cat._by_id.get(t.object_id, (None,))[0]
    return sl, ol


def normalize_predicates(triples: list[C.ExtractedTriple], cat: Catalog,
                         cfg: ExtractConfig, emb=None, llm=None) -> list[C.ExtractedTriple]:
    mode = getattr(cfg, "normalize_mode", "off")
    if mode == "off" or not triples:
        return triples
    if mode == "endpoint_override":
        return _normalize_endpoint(triples, cat)
    if mode == "embed_cluster":
        if emb is None:
            print("  [warn] normalize_mode=embed_cluster 인데 EmbeddingClient 없음 -> 정규화 skip")
            return triples
        return _normalize_embed(triples, cat, emb, cfg)
    if mode == "guarded_override":
        if emb is None:
            print("  [warn] normalize_mode=guarded_override 인데 EmbeddingClient 없음 -> 정규화 skip")
            return triples
        return _normalize_guarded(triples, cat, emb, cfg)
    if mode == "llm_sense":
        if llm is None:
            print("  [warn] normalize_mode=llm_sense 인데 LLMClient 없음 -> 정규화 skip")
            return triples
        return _normalize_llm_sense(triples, cat, llm, cfg)
    print(f"  [warn] 알 수 없는 normalize_mode={mode!r} -> skip")
    return triples


def _unit(v: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _normalize_guarded(triples: list[C.ExtractedTriple], cat: Catalog,
                       emb, cfg: ExtractConfig) -> list[C.ExtractedTriple]:
    """방법 1.5: 알려진 라벨쌍의 sense 인벤토리에 predicate를 임베딩 코사인으로 매핑.
    argmax sense 의 코사인 >= sense_tau (그리고 top1-top2 >= sense_margin)면 그 정준명으로,
    아니면 어디에도 애매한 것으로 보고 원본 predicate 유지(reject)."""
    groups: dict[frozenset, list[C.ExtractedTriple]] = defaultdict(list)
    for t in triples:
        sl, ol = _labels(cat, t)
        if sl is None or ol is None:
            continue
        pair = frozenset({sl, ol})
        if pair in _ENDPOINT_SENSES:           # sense 인벤토리 있는 쌍만 대상, 나머지는 원본 유지
            groups[pair].append(t)

    total = 0
    for pair, ts in groups.items():
        senses = _ENDPOINT_SENSES[pair]
        labs = sorted(pair)
        a, b = labs[0], (labs[1] if len(labs) > 1 else labs[0])
        # distinct predicate + 대표 cue
        cue: dict[str, str] = {}
        preds: list[str] = []
        for t in ts:
            if t.predicate not in cue:
                cue[t.predicate] = t.predicate_cue or ""
                preds.append(t.predicate)
        proto_names = [s[0] for s in senses]
        proto_texts = [s[1] for s in senses]
        pred_texts = [f"{a} {p.replace('_', ' ').lower()} {b}. e.g., {cue[p]}" for p in preds]
        try:
            vecs = emb.embed(proto_texts + pred_texts)
        except Exception as e:
            print(f"  [warn] guarded 임베딩 실패({type(e).__name__}) -> ({a},{b}) skip")
            continue
        if len(vecs) != len(proto_texts) + len(pred_texts):
            print(f"  [warn] guarded 임베딩 개수 불일치 -> ({a},{b}) skip")
            continue
        pv = [_unit(v) for v in vecs[:len(senses)]]
        xv = [_unit(v) for v in vecs[len(senses):]]
        pmap: dict[str, str] = {}
        for i, p in enumerate(preds):
            sims = [sum(x * y for x, y in zip(xv[i], pv[k])) for k in range(len(senses))]
            order = sorted(range(len(senses)), key=lambda k: sims[k], reverse=True)
            best = order[0]
            second = sims[order[1]] if len(order) > 1 else -1.0
            ok = sims[best] >= cfg.sense_tau and (sims[best] - second) >= cfg.sense_margin
            pmap[p] = proto_names[best] if ok else p
            tag = pmap[p] if ok else "(keep)"
            print(f"  [guarded] ({a},{b}) {p!r} sim={sims[best]:.2f} -> {tag}")
        for t in ts:
            newp = pmap.get(t.predicate, t.predicate)
            if newp != t.predicate:
                t.predicate = newp
                total += 1
    print(f"  [normalize:guarded] {total}개 트리플 재명명(sense_tau={cfg.sense_tau}, margin={cfg.sense_margin})")
    return triples


_SENSE_SYSTEM = (
    "You label ONE relationship between two manufacturing/supply-chain entities into exactly one "
    "of the given sense categories, judging ONLY from the evidence sentence. Read it carefully — "
    "direction and cause/effect matter (e.g. 'a PO expedited to fix an anomaly' is the OPPOSITE of "
    "'a defective delivery that caused the anomaly'). If none of the categories clearly fits, "
    "answer OTHER. Output JSON only: {\"sense\": \"<CATEGORY_NAME or OTHER>\"}."
)
_SENSE_SCHEMA = {
    "type": "object",
    "properties": {"sense": {"type": "string"}},
    "required": ["sense"],
}


def _normalize_llm_sense(triples: list[C.ExtractedTriple], cat: Catalog, llm,
                         cfg: ExtractConfig) -> list[C.ExtractedTriple]:
    """방법 1(LLM sense 분류): 아는 브리지쌍 엣지의 sense를 임베딩 코사인이 아니라 chat-LLM이
    evidence 문장을 읽고 후보 sense 중 택1(또는 OTHER→원본 유지)로 결정한다. 약한 임베더가
    'RELATED_TO' 같은 껍데기 술어로 추측하던 것을, 강한 LLM이 원문 문장으로 분류하게 바꿈.
    evidence 문장 기준으로 캐시(동일 문장은 1회만 호출)."""
    struct_map = structured_pair_senses()   # 정형 FK 후보(동적 fold)
    struct_dir = structured_rel_dir()       # 정형 관계의 정준 방향(fold 시 재정렬용)
    total, calls, forced = 0, 0, 0
    cache: dict[tuple, str] = {}          # (pair, evidence) -> 결정된 sense명 or "__KEEP__"
    for t in triples:
        sl, ol = _labels(cat, t)
        if sl is None or ol is None:
            continue
        pair = frozenset({sl, ol})
        struct = struct_map.get(pair, [])         # 정형 관계 후보(있으면 FK-fold 대상)
        hand = _ENDPOINT_SENSES.get(pair, [])     # 수기 브리지 sense
        senses = struct + hand                    # 정형 후보 먼저, 그다음 수기 sense
        if not senses:                    # 정형도 수기도 없는 쌍 = 진짜 doc-only 파편 -> 원본 유지
            continue
        if len(senses) == 1 and not struct:  # 순수 수기 단일-sense: recall 우선 강제할당(기존 동작 유지)
            chosen = senses[0][0]
            if chosen != t.predicate:
                t.predicate = chosen
                total += 1
            forced += 1
            continue
        # 정형 후보 포함 or 다의 => LLM이 evidence로 택1(정형 관계면 fold) 또는 OTHER(다르면 raw 유지)
        ev = (t.predicate_cue or "").strip()
        key = (pair, ev)
        if key not in cache:
            labs = sorted(pair)
            a, b = labs[0], (labs[1] if len(labs) > 1 else labs[0])
            menu = "\n".join(f"- {n}: {d}" for (n, d) in senses) + "\n- OTHER: none of the above."
            user = (f"Entities: ({a}) and ({b}).\n\nCandidate senses:\n{menu}\n\n"
                    f"Evidence sentence: \"{ev}\"\n\nWhich single sense does the sentence state? "
                    f"Return {{\"sense\": \"...\"}}.")
            try:
                out = llm.chat_json(_SENSE_SYSTEM, user, _SENSE_SCHEMA, max_tokens=60)
                s = str(out.get("sense", "")).strip().upper()
            except Exception as e:
                print(f"  [warn] llm_sense 분류 실패({type(e).__name__}) -> keep raw")
                s = ""
            valid = {n.upper() for (n, _) in senses}
            cache[key] = s if s in valid else "__KEEP__"
            calls += 1
            print(f"  [llm_sense] ({a},{b}) {ev[:60]!r} -> {cache[key]}")
        chosen = cache[key]
        if chosen != "__KEEP__":
            _dk = (pair, chosen)
            if _dk in struct_dir:                         # 정형명으로 fold -> 정형 방향으로 재정렬
                want_src = struct_dir[_dk][0]
                if cat._by_id.get(t.subject_id, (None,))[0] != want_src:
                    t.subject_id, t.object_id = t.object_id, t.subject_id
                    t.subject_match, t.object_match = t.object_match, t.subject_match
                    t.subject_surface, t.object_surface = t.object_surface, t.subject_surface
            if chosen != t.predicate:
                t.predicate = chosen
                total += 1
    print(f"  [normalize:llm_sense] {total}개 트리플 재명명 "
          f"(단일-sense 강제 {forced}회, 다중-sense LLM 호출 {calls}회)")
    return triples


def _normalize_endpoint(triples: list[C.ExtractedTriple], cat: Catalog) -> list[C.ExtractedTriple]:
    """방법 1: 아는 3쌍이면 정준 predicate로 무조건 덮어씀(안전장치 없음)."""
    n = 0
    for t in triples:
        sl, ol = _labels(cat, t)
        if sl is None or ol is None:
            continue
        canon = _ENDPOINT_CANON.get(frozenset({sl, ol}))
        if canon and t.predicate != canon:
            t.predicate = canon
            n += 1
    print(f"  [normalize:endpoint] {n}개 트리플을 allowlist 정준 predicate로 덮어씀")
    return triples


def _merge_extracted_embed(triples: list[C.ExtractedTriple], cat: Catalog,
                           emb, cfg: ExtractConfig) -> list[C.ExtractedTriple]:
    """③(옵션) open 모드 비-백본 :ExtractedEntity 를 표면형 임베딩 코사인으로 병합(tau=entity_tau).
    ①백본 사상·②정준키 dedup 이후 남은 표면 변형(동의어)만 추가 병합. 기본 off, --entity-merge embed."""
    support: dict[str, int] = defaultdict(int)
    for t in triples:
        for sid in (t.subject_id, t.object_id):
            if cat._by_id.get(sid, (None,))[0] == cat.EXT_LABEL:
                support[sid] += 1
    ids = list(support)
    if len(ids) < 2:
        return triples
    texts = [cat._by_id[i][1] for i in ids]       # 대표 surface
    try:
        vecs = emb.embed(texts)
    except Exception as e:
        print(f"  [warn] entity 임베딩 실패({type(e).__name__}) -> 병합 skip")
        return triples
    canon = _cluster_to_canonical(ids, vecs, support, cfg.entity_tau)   # id -> 정준 id
    nmerge = sum(1 for i, c in canon.items() if c != i)
    for t in triples:
        t.subject_id = canon.get(t.subject_id, t.subject_id)
        t.object_id = canon.get(t.object_id, t.object_id)
    print(f"  [entity:embed] ExtractedEntity {nmerge}개 병합(tau={cfg.entity_tau})")
    return triples


def _normalize_embed(triples: list[C.ExtractedTriple], cat: Catalog,
                     emb, cfg: ExtractConfig) -> list[C.ExtractedTriple]:
    """방법 2A: 라벨쌍 내에서 predicate를 임베딩 코사인으로 군집 -> 정준화."""
    groups: dict[frozenset, list[C.ExtractedTriple]] = defaultdict(list)
    for t in triples:
        sl, ol = _labels(cat, t)
        if sl is None or ol is None:
            continue
        groups[frozenset({sl, ol})].append(t)

    total = 0
    for pair, ts in groups.items():
        support: dict[str, int] = defaultdict(int)
        cue: dict[str, str] = {}
        for t in ts:
            support[t.predicate] += 1
            if t.predicate not in cue and t.predicate_cue:
                cue[t.predicate] = t.predicate_cue
        preds = list(support.keys())
        if len(preds) < 2:                       # 병합할 후보가 없음
            continue
        labs = sorted(pair)
        a, b = labs[0], (labs[1] if len(labs) > 1 else labs[0])
        texts = [f"{a} {p.replace('_', ' ').lower()} {b}. e.g., {cue.get(p, '')}" for p in preds]
        try:
            vecs = emb.embed(texts)
        except Exception as e:
            print(f"  [warn] 임베딩 호출 실패({type(e).__name__}) -> ({a},{b}) skip")
            continue
        canon_map = _cluster_to_canonical(preds, vecs, support, cfg.emb_tau)
        merged: dict[str, list[str]] = defaultdict(list)
        for p, c in canon_map.items():
            merged[c].append(p)
        for c, members in merged.items():
            if len(members) > 1:
                print(f"  [normalize:embed] ({a},{b}) {sorted(members)} -> {c}")
        for t in ts:
            newp = canon_map.get(t.predicate, t.predicate)
            if newp != t.predicate:
                t.predicate = newp
                total += 1
    print(f"  [normalize:embed] 총 {total}개 트리플 재명명(tau={cfg.emb_tau})")
    return triples


def _cluster_to_canonical(preds: list[str], vecs: list[list[float]],
                          support: dict[str, int], tau: float) -> dict[str, str]:
    """코사인>=tau 를 잇는 연결요소로 군집. 성분 정준 = 최다지지(동률이면 최단명)."""
    n = len(preds)
    if n != len(vecs) or n == 0:                 # 임베딩 개수 불일치 방어
        return {p: p for p in preds}
    unit = []
    for v in vecs:
        s = math.sqrt(sum(x * x for x in v)) or 1.0
        unit.append([x / s for x in v])

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    for i in range(n):
        for j in range(i + 1, n):
            cos = sum(x * y for x, y in zip(unit[i], unit[j]))
            if cos >= tau:
                union(i, j)

    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    canon_map: dict[str, str] = {}
    for idxs in comp.values():
        best = max(idxs, key=lambda i: (support[preds[i]], -len(preds[i])))
        for i in idxs:
            canon_map[preds[i]] = preds[best]
    return canon_map


def run_extraction(documents: list[dict], cat: Catalog, llm, cfg: ExtractConfig | None = None,
                   emb=None):
    """전체 문서 추출 -> (승격된 WriteOp 리스트, 전체 ScoreBreakdown 리스트, FunnelStats)."""
    cfg = cfg or ExtractConfig()
    funnel = FunnelStats()
    chunks = chunk_corpus(documents, size=cfg.window_size, step=cfg.window_step)
    all_triples: list[C.ExtractedTriple] = []
    for ch in chunks:
        all_triples.extend(_extract_chunk(ch, cat, llm, cfg, funnel))

    # Entity 축 B(open 모드) ③ 임베딩 병합(옵션). ①백본·②정준키는 _extract_chunk 에서 이미 반영.
    if (getattr(cfg, "entities", "bound") == "open"
            and getattr(cfg, "entity_merge", "string") == "embed" and emb is not None):
        all_triples = _merge_extracted_embed(all_triples, cat, emb, cfg)

    funnel.predicates_before = len({t.predicate for t in all_triples})   # 정규화 전 파편화
    all_triples = normalize_predicates(all_triples, cat, cfg, emb, llm)
    funnel.predicates_after = len({t.predicate for t in all_triples})    # 정규화 후

    # predicate 파편화 감사: 인벤토리 = 수기 브리지 sense ∪ 정형 관계명(FK-fold 결과 포함)
    _struct = structured_pair_senses()
    _struct_pairs = set(_struct.keys())
    _struct_names = {n for lst in _struct.values() for (n, _) in lst}
    _inv = set(SENSE_GLOSS) | _struct_names
    funnel.inventory_names = _inv
    funnel.predicate_counts = Counter(t.predicate for t in all_triples)
    _pair = defaultdict(Counter)
    for t in all_triples:
        if t.predicate not in _inv:                       # 인벤토리 밖 = 아직 정규화 안 된 doc-only 파편
            sl, ol = _labels(cat, t)
            _pair[f"{sl}->{ol}"][t.predicate] += 1
    funnel.pair_residual = {k: v.most_common() for k, v in    # 잔존 2종 이상인 쌍만(합칠 게 있는 쌍)
                            sorted(_pair.items(), key=lambda kv: -len(kv[1])) if len(v) >= 2}

    # genuine-bridge 분할: 정형 label쌍이 '없는' 엣지 = 테이블로 표현 불가한 연결(=C1 통합필연성의 직접 정량증거)
    _bridge_types, n_bridge, n_fk = set(), 0, 0
    for t in all_triples:
        sl, ol = _labels(cat, t)
        if sl is None or ol is None:
            continue
        if frozenset({sl, ol}) in _struct_pairs:
            n_fk += 1
        else:
            n_bridge += 1
            _bridge_types.add(t.predicate)
    funnel.genuine_bridge_edges = n_bridge
    funnel.genuine_bridge_types = len(_bridge_types)
    funnel.fk_redundant_edges = n_fk

    breakdowns = C.score_all(all_triples, cfg.conf)

    ops: list[WriteOp] = []
    for b in breakdowns:
        if not b.promoted:
            continue
        t = b.triple
        s_label, s_name = cat._by_id[t.subject_id]
        o_label, o_name = cat._by_id[t.object_id]
        ops.append(WriteOp(
            src_label=s_label, src_id=t.subject_id, src_pk=cat.label_pk[s_label],
            rel_type=t.predicate,
            tgt_label=o_label, tgt_id=t.object_id, tgt_pk=cat.label_pk[o_label],
            props={"origin": "extracted", "confidence": round(b.confidence, 3),
                   "source_doc": t.source_doc_id, "source_chunk_id": t.source_chunk_id,
                   "evidence": t.predicate_cue},
            src_name=(s_name if s_label == cat.EXT_LABEL else None),
            tgt_name=(o_name if o_label == cat.EXT_LABEL else None),
        ))
    return ops, breakdowns, funnel


# ---------------------------------------------------------------------------
# 쓰기 (Neo4j) — 검색 경로와 분리된 쓰기 전용 경로
# ---------------------------------------------------------------------------
def write_ops(driver, database: str, ops: list[WriteOp]) -> int:
    """승격 엣지 적재. 같은 (src)-[rel]->(tgt)는 MERGE로 idempotent. 재실행 안전."""
    # (rel_type, src_label, src_pk, tgt_label, tgt_pk)별 배치
    buckets: dict[tuple, list] = {}
    for op in ops:
        buckets.setdefault((op.rel_type, op.src_label, op.src_pk, op.tgt_label, op.tgt_pk), []).append(
            {"s": op.src_id, "t": op.tgt_id, "props": op.props,
             "sn": op.src_name, "tn": op.tgt_name})

    def _node(var: str, label: str, pk: str, id_key: str, name_key: str) -> str:
        # :ExtractedEntity(open 모드)는 그래프에 없으므로 MERGE로 생성, 백본은 MATCH(전제 존재)
        if label == Catalog.EXT_LABEL:
            return (f"MERGE ({var}:`{label}` {{`{pk}`: row.{id_key}}}) "
                    f"ON CREATE SET {var}.name = coalesce(row.{name_key}, row.{id_key}), "
                    f"{var}.origin = 'extracted' ")
        return f"MATCH ({var}:`{label}` {{`{pk}`: row.{id_key}}}) "

    n = 0
    for (rel, sl, spk, tl, tpk), rows in buckets.items():
        specs = [("a", sl, spk, "s", "sn"), ("b", tl, tpk, "t", "tn")]
        # Cypher는 MERGE 다음 MATCH에 WITH를 요구 => MATCH(백본)를 먼저, MERGE(:ExtractedEntity)를 뒤로 배치.
        matches = [_node(*s) for s in specs if s[1] != Catalog.EXT_LABEL]
        merges = [_node(*s) for s in specs if s[1] == Catalog.EXT_LABEL]
        cypher = (                                   # FK-fold이 정형 엣지 위에 MERGE될 때 origin 덮어쓰기 방지:
            "UNWIND $rows AS row "                    #  - 새 엣지(정형 대응 없음=genuine bridge) => 정상 write(origin=extracted)
            + "".join(matches) + "".join(merges)      #  - 기존 엣지와 겹침(FK-중복) => origin 보존, 문서 corroboration만 표시
            + f"MERGE (a)-[r:`{rel}`]->(b) "
            + "ON CREATE SET r += row.props "
            + "ON MATCH SET r.doc_corroborations = coalesce(r.doc_corroborations, 0) + 1"
        )
        driver.execute_query(cypher, rows=rows, database_=database)
        n += len(rows)
    return n


def create_chunk_graph(driver, database: str, documents: list[dict], cfg: ExtractConfig):
    """(Document)-[:HAS_CHUNK]->(Chunk) 적재 + Chunk.text full-text 인덱스."""
    chunks = chunk_corpus(documents, size=cfg.window_size, step=cfg.window_step)
    rows = [c.as_dict() for c in chunks]
    driver.execute_query(
        "UNWIND $rows AS row "
        "MERGE (c:`Chunk` {chunk_id: row.chunk_id}) "
        "SET c.doc_id=row.doc_id, c.seq=row.seq, c.text=row.text "
        "WITH c, row MATCH (d:`Document` {document_id: row.doc_id}) "
        "MERGE (d)-[:HAS_CHUNK]->(c)",
        rows=rows, database_=database)
    driver.execute_query(
        "CREATE FULLTEXT INDEX chunkText IF NOT EXISTS FOR (c:`Chunk`) ON EACH [c.text]",
        database_=database)
    return len(rows)