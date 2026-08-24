"""
build_graph_neo4j.py  (v3 — KG-RAG Mother Graph, deterministic backbone, graceful-skip)
---------------------------------------------------------------------------------------
결정 반영:
  1) 정형 KG + document_links '결정적 backbone'까지만. LLM 본문 추출 엣지는 제외(다음 단계).
  2) 필요한 테이블만. 단, '있는 파일만 적재'(graceful skip)로 확장 — 파일을 점진적으로
     올려도 그대로 동작. 없는 파일/노드를 가리키는 관계는 조용히 건너뛰고 리포트에 표기.
  3) document_links 는 relationship 값을 '타입형' 엣지로 승격(investigates -> INVESTIGATES).
  4) origin 태깅: 'structured' | 'document_link' | ('extracted'는 다음 단계).
  5) product_suppliers : (Product)-[:SUPPLIED_BY {계약속성}]->(Supplier).
  6) WIPE_FIRST=True.

파일 세트(있는 것만 적재):
  [핵심 7]  products, suppliers, product_suppliers, anomaly_events, downtime_events,
            documents, document_links
  [A backbone 대상]  service_tickets, purchase_orders, contracts
  [B spine 연결]     customers, purchase_order_lines, regions
  [C 설비 spine]     equipment, lines

정직성 메모:
  - 파일이 없으면 해당 노드/관계는 생성하지 않고 '건너뜀'으로 리포트(행을 버리는 게 아님).
  - document_links 의 대상 노드가 아직 없으면 '보류'로 집계. 대상 테이블을 올리면 자동 연결.
"""

import os
import re
import math
import pandas as pd
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------------------------

URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DATA_DIR = os.getenv("DATA_DIR", "data")

# 문서 소스(교체됨): 기존 documents.csv/document_links.csv -> *_new. env로 되돌리기 가능.
DOCUMENTS_FILE  = os.getenv("DOCUMENTS_FILE",  "documents.csv")
DOC_LINKS_FILE  = os.getenv("DOC_LINKS_FILE",  "document_links.csv")

WIPE_FIRST = True

SENTINELS = {"ALL", "All", "all"}

DATETIME_COLS = {"event_time", "start_time", "end_time", "opened_at", "closed_at"}
DATE_COLS = {
    "document_date", "valid_from", "valid_to", "effective_date",
    "order_date", "expected_date", "start_date", "end_date",
}

# ---------------------------------------------------------------------------
# 1. 노드 스펙: (파일, 라벨, PK)  — 있는 파일만 실제 적재
# ---------------------------------------------------------------------------
NODE_SPECS = [
    ("products.csv",         "Product",       "product_id"),
    ("suppliers.csv",        "Supplier",      "supplier_id"),
    ("anomaly_events.csv",   "AnomalyEvent",  "anomaly_id"),
    ("downtime_events.csv",  "DowntimeEvent", "downtime_id"),
    (DOCUMENTS_FILE,         "Document",      "document_id"),
    # --- A: backbone 대상 ---
    ("service_tickets.csv",  "ServiceTicket", "ticket_id"),
    ("purchase_orders.csv",  "PurchaseOrder", "purchase_order_id"),
    ("contracts.csv",        "Contract",      "contract_id"),
    # --- B: spine 연결 ---
    ("customers.csv",        "Customer",      "customer_id"),
    ("regions.csv",          "Region",        "region_id"),
    # --- C: 설비 spine ---
    ("equipment.csv",        "Equipment",     "equipment_id"),
    ("lines.csv",            "Line",          "line_id"),
]

# ---------------------------------------------------------------------------
# 2. 일반 관계 스펙(소스 '행=노드'의 FK 엣지). 전부 origin='structured'.
#    (파일, 소스라벨, 소스PK, FK컬럼, 타겟라벨, 타겟PK, 관계타입)
# ---------------------------------------------------------------------------
REL_SPECS = [
    # 설비/제품 spine
    ("equipment.csv",       "Equipment",     "equipment_id", "line_id",            "Line",         "line_id",     "ON_LINE"),
    ("products.csv",        "Product",       "product_id",   "line_id",            "Line",         "line_id",     "ON_LINE"),
    ("anomaly_events.csv",  "AnomalyEvent",  "anomaly_id",   "equipment_id",       "Equipment",    "equipment_id","ON_EQUIPMENT"),
    ("downtime_events.csv", "DowntimeEvent", "downtime_id",  "equipment_id",       "Equipment",    "equipment_id","ON_EQUIPMENT"),
    # 이상/가동중단 체인
    ("downtime_events.csv", "DowntimeEvent", "downtime_id",  "anomaly_id",         "AnomalyEvent", "anomaly_id",  "LINKED_ANOMALY"),
    ("downtime_events.csv", "DowntimeEvent", "downtime_id",  "affected_product_id","Product",      "product_id",  "AFFECTED_PRODUCT"),
    # 고객/지역
    ("customers.csv",       "Customer",      "customer_id",  "region_id",          "Region",       "region_id",   "IN_REGION"),
    # 서비스 티켓
    ("service_tickets.csv", "ServiceTicket", "ticket_id",    "customer_id",        "Customer",     "customer_id", "FROM_CUSTOMER"),
    ("service_tickets.csv", "ServiceTicket", "ticket_id",    "product_id",         "Product",      "product_id",  "ABOUT_PRODUCT"),
    # 구매발주
    ("purchase_orders.csv", "PurchaseOrder", "purchase_order_id", "supplier_id",   "Supplier",     "supplier_id", "FROM_SUPPLIER"),
    # 계약
    ("contracts.csv",       "Contract",      "contract_id",  "customer_id",        "Customer",     "customer_id", "FOR_CUSTOMER"),
]

# 정션 테이블(행=관계+속성)용 스펙: (파일, 소스라벨, 소스FK, 타겟라벨, 타겟FK, 관계타입, drop컬럼들)
JUNCTION_SPECS = [
    ("product_suppliers.csv",     "Product",       "product_id",
     "Supplier", "supplier_id", "SUPPLIED_BY",   ["product_id", "supplier_id"]),
    ("purchase_order_lines.csv",  "PurchaseOrder", "purchase_order_id",
     "Product",  "product_id",  "ORDERS_PRODUCT", ["purchase_order_id", "product_id", "po_line_id"]),
]

# document_links.entity_type -> (라벨, PK)
ENTITY_TYPE_LABEL = {
    "anomaly":        ("AnomalyEvent",  "anomaly_id"),
    "service_ticket": ("ServiceTicket", "ticket_id"),
    "purchase_order": ("PurchaseOrder", "purchase_order_id"),
    "contract":       ("Contract",      "contract_id"),
}


# ---------------------------------------------------------------------------
# 3. 헬퍼
# ---------------------------------------------------------------------------
def _path(name): return os.path.join(DATA_DIR, name)
def exists(name): return os.path.isfile(_path(name))
def read_csv(name): return pd.read_csv(_path(name), encoding="utf-8-sig")


def clean_value(col, val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if col in DATETIME_COLS:
        return pd.to_datetime(val).to_pydatetime()
    if col in DATE_COLS:
        return pd.to_datetime(val).date()
    if hasattr(val, "item"):
        return val.item()
    return val


def row_to_props(record):
    props = {}
    for col, val in record.items():
        cv = clean_value(col, val)
        if cv is not None and cv != "":
            props[col] = cv
    return props


def rel_type_of(relationship: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(relationship).strip()).strip("_").upper()
    return t or "LINKS"


def loaded_labels() -> set:
    return {label for fn, label, _ in NODE_SPECS if exists(fn)}


# ---------------------------------------------------------------------------
# 4. 적재
# ---------------------------------------------------------------------------
def create_constraints(driver):
    n = 0
    for fn, label, pk in NODE_SPECS:
        if not exists(fn):
            continue
        driver.execute_query(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.`{pk}` IS UNIQUE",
            database_=DATABASE)
        n += 1
    print(f"  제약조건 {n}개")


def load_nodes(driver):
    print("\n[노드 적재]")
    for fn, label, pk in NODE_SPECS:
        if not exists(fn):
            print(f"  {label:<14}   -- (파일 없음: {fn}) 건너뜀")
            continue
        df = read_csv(fn)
        rows = [row_to_props(r) for r in df.to_dict("records")]
        driver.execute_query(
            f"UNWIND $rows AS row MERGE (n:`{label}` {{`{pk}`: row.`{pk}`}}) SET n += row",
            rows=rows, database_=DATABASE)
        print(f"  {label:<14} {len(rows):>5} 노드")


def load_relationships(driver):
    print("\n[관계 적재 · structured (일반 FK)]")
    have = loaded_labels()
    for fn, sl, spk, fk, tl, tpk, rt in REL_SPECS:
        if not exists(fn) or sl not in have or tl not in have:
            reason = "파일없음" if not exists(fn) else ("타깃없음:" + tl if tl not in have else "소스없음:" + sl)
            print(f"  ({sl})-[:{rt}]->({tl})  -- 건너뜀({reason})")
            continue
        df = read_csv(fn)
        pairs, skipped = [], 0
        for rec in df.to_dict("records"):
            src = clean_value(spk, rec.get(spk)); tgt = rec.get(fk)
            if tgt is None or (isinstance(tgt, float) and math.isnan(tgt)) \
               or str(tgt).strip() in SENTINELS or str(tgt).strip() == "":
                skipped += 1; continue
            pairs.append({"src": src, "tgt": clean_value(fk, tgt)})
        driver.execute_query(
            f"UNWIND $rows AS row "
            f"MATCH (a:`{sl}` {{`{spk}`: row.src}}) MATCH (b:`{tl}` {{`{tpk}`: row.tgt}}) "
            f"MERGE (a)-[r:`{rt}`]->(b) SET r.origin='structured'",
            rows=pairs, database_=DATABASE)
        tag = f"(널/센티넬 {skipped} skip)" if skipped else ""
        print(f"  ({sl})-[:{rt}]->({tl})  {len(pairs):>5} 엣지 {tag}")


def load_junctions(driver):
    print("\n[관계 적재 · junction 테이블]")
    have = loaded_labels()
    for fn, sl, sfk, tl, tfk, rt, drop in JUNCTION_SPECS:
        if not exists(fn) or sl not in have or tl not in have:
            reason = "파일없음" if not exists(fn) else ("타깃없음:" + tl if tl not in have else "소스없음:" + sl)
            print(f"  ({sl})-[:{rt}]->({tl})  -- 건너뜀({reason})")
            continue
        df = read_csv(fn)
        rows = []
        for rec in df.to_dict("records"):
            s = rec.get(sfk); t = rec.get(tfk)
            if not s or not t:
                continue
            props = row_to_props(rec)
            for c in drop:
                props.pop(c, None)
            rows.append({"s": clean_value(sfk, s), "t": clean_value(tfk, t), "props": props})
        driver.execute_query(
            f"UNWIND $rows AS row "
            f"MATCH (a:`{sl}` {{`{sfk}`: row.s}}) MATCH (b:`{tl}` {{`{tfk}`: row.t}}) "
            f"MERGE (a)-[r:`{rt}`]->(b) SET r += row.props, r.origin='structured'",
            rows=rows, database_=DATABASE)
        print(f"  ({sl})-[:{rt}]->({tl})  {len(rows):>5} 엣지")


def load_document_links(driver):
    print("\n[관계 적재 · document_links (backbone)]")
    if not exists(DOC_LINKS_FILE):
        print(f"  -- {DOC_LINKS_FILE} 없음, 건너뜀"); return
    have = loaded_labels()
    df = read_csv(DOC_LINKS_FILE)
    buckets, deferred = {}, {}
    for rec in df.to_dict("records"):
        et = str(rec.get("entity_type", "")).strip()
        m = ENTITY_TYPE_LABEL.get(et)
        if m is None:
            deferred[f"unknown:{et}"] = deferred.get(f"unknown:{et}", 0) + 1; continue
        tl, tpk = m
        if tl not in have:
            deferred[et] = deferred.get(et, 0) + 1; continue
        rt = rel_type_of(rec.get("relationship"))
        try: conf = float(rec.get("confidence"))
        except (TypeError, ValueError): conf = None
        buckets.setdefault((rt, tl, tpk), []).append({
            "doc": rec.get("document_id"), "tgt": rec.get("entity_id"),
            "props": {"relationship": rec.get("relationship"), "entity_type": et,
                      "excerpt": rec.get("extracted_excerpt"), "confidence": conf,
                      "origin": "document_link"}})
    total = 0
    for (rt, tl, tpk), rows in sorted(buckets.items()):
        driver.execute_query(
            f"UNWIND $rows AS row "
            f"MATCH (d:`Document` {{document_id: row.doc}}) MATCH (t:`{tl}` {{`{tpk}`: row.tgt}}) "
            f"MERGE (d)-[r:`{rt}`]->(t) SET r += row.props",
            rows=rows, database_=DATABASE)
        total += len(rows)
        print(f"  (Document)-[:{rt}]->({tl})  {len(rows):>5} 엣지")
    if deferred:
        print("  보류(타깃 노드 미적재 — 해당 테이블 올리면 연결됨):")
        for k, v in sorted(deferred.items()):
            print(f"    {k:<18} {v}")
    print(f"  backbone 총 {total} 엣지")


def wipe(driver):
    driver.execute_query("MATCH (n) DETACH DELETE n", database_=DATABASE)
    print("  기존 그래프 삭제")


def report(driver):
    print("\n[검증 요약]")
    n, _, _ = driver.execute_query("MATCH (n) RETURN count(n) AS c", database_=DATABASE)
    r, _, _ = driver.execute_query("MATCH ()-[x]->() RETURN count(x) AS c", database_=DATABASE)
    print(f"  총 노드 {n[0]['c']} / 총 엣지 {r[0]['c']}")
    rows, _, _ = driver.execute_query(
        "MATCH ()-[x]->() RETURN type(x) AS t, x.origin AS o, count(*) AS c ORDER BY t,o",
        database_=DATABASE)
    for row in rows:
        print(f"    {row['t']:<16} origin={row['o'] or '-':<13} {row['c']}")


def main():
    print(f"Neo4j 접속: {URI} / db={DATABASE} / data={DATA_DIR}")
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        driver.verify_connectivity()
        if WIPE_FIRST:
            wipe(driver)
        create_constraints(driver)
        load_nodes(driver)
        load_relationships(driver)
        load_junctions(driver)
        load_document_links(driver)
        report(driver)
    print("\n완료.")


if __name__ == "__main__":
    main()
