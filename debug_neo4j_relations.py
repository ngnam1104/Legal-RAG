"""
debug_neo4j_relations.py
========================
Phân tích relationship types trong Neo4j và mô phỏng hiệu quả của normalization mới.

Chạy: python debug_neo4j_relations.py [--uri bolt://localhost:7688] [--top 30]
"""
import os
import sys
import re
import argparse
from collections import defaultdict
from dotenv import load_dotenv

# Fix UTF-8 cho Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[ERROR] neo4j driver chua cai: pip install neo4j")
    sys.exit(1)

# =====================================================================
# Inline normalization constants (tranh import entities.py vi no keo
# theo icllmlib / LLM client khong can thiet cho debug script nay)
# =====================================================================
FIXED_NODE_RELATIONS = {
    "ISSUED_BY", "SIGNED_BY", "APPROVED_BY", "PUBLISHED_BY",
    "CREATED_BY", "ESTABLISHED_BY",
    "IMPLEMENTED_BY", "ENFORCED_BY", "APPLIED_BY", "EXECUTED_BY",
    "MANAGED_BY", "REGULATED_BY", "COORDINATED_BY",
    "TRANSFERRED_TO", "TRANSFERRED_FROM", "SUBMITTED_TO",
    "DELEGATED_TO", "ASSIGNED_TO", "ASSIGNED_BY",
    "REPORTED_TO", "NOTIFIED_TO",
    "PERMITTED_TO", "PROHIBITED_FROM", "EXEMPT_FROM", "ENTITLED_TO",
    "REQUIRED_FOR", "REQUIRED_BY", "COMPLIES_WITH", "AFFECTED_BY",
    "DEFINED_IN", "CLASSIFIED_AS", "BELONGS_TO", "PART_OF",
    "LOCATED_IN", "MEMBER_OF",
    "FUNDED_BY", "PAID_TO", "PAID_BY", "COLLECTED_BY",
    "REPLACED_BY", "AMENDED_BY", "REPEALED_BY", "REFERENCED_BY", "BASED_ON",
    "APPLIES_TO", "RELATED_TO",
}


_VERB_ROOT_CANONICAL = {
    "ISSUE": "ISSUED_BY", "SIGN": "SIGNED_BY", "APPROV": "APPROVED_BY",
    "PUBLISH": "PUBLISHED_BY", "CREAT": "CREATED_BY", "ESTABLISH": "ESTABLISHED_BY",
    "IMPLEMENT": "IMPLEMENTED_BY", "ENFORC": "ENFORCED_BY",
    "APPLY": "APPLIED_BY", "APPLI": "APPLIED_BY", "EXECUT": "EXECUTED_BY",
    "PERFORM": "IMPLEMENTED_BY", "CARRY": "IMPLEMENTED_BY",
    "MANAG": "MANAGED_BY", "GOVERN": "MANAGED_BY",
    "SUPERVIS": "MANAGED_BY", "DIRECT": "MANAGED_BY",
    "REGULAT": "REGULATED_BY", "COORDINAT": "COORDINATED_BY",
    "TRANSFER": "TRANSFERRED_TO", "SUBMIT": "SUBMITTED_TO",
    "DELEGAT": "DELEGATED_TO", "ASSIGN": "ASSIGNED_TO",
    "REPORT": "REPORTED_TO", "NOTIF": "NOTIFIED_TO", "INFORM": "NOTIFIED_TO",
    "PERMIT": "PERMITTED_TO", "PROHIBIT": "PROHIBITED_FROM",
    "EXEMPT": "EXEMPT_FROM", "ENTITL": "ENTITLED_TO",
    "REQUIR": "REQUIRED_FOR", "AFFECT": "AFFECTED_BY",
    "DEFIN": "DEFINED_IN", "CLASSIF": "CLASSIFIED_AS",
    "BELONG": "BELONGS_TO", "FUND": "FUNDED_BY",
    "PAY": "PAID_TO", "COLLECT": "COLLECTED_BY",
    "REPLAC": "REPLACED_BY", "AMEND": "AMENDED_BY",
    "REPEAL": "REPEALED_BY", "REFERENC": "REFERENCED_BY",
    "RELAT": "RELATED_TO",
}

_ALIAS = {
    "ISSUES": "ISSUED_BY", "ISSUED": "ISSUED_BY",
    "SIGNS": "SIGNED_BY", "SIGNED": "SIGNED_BY",
    "APPROVES": "APPROVED_BY", "APPROVED": "APPROVED_BY",
    "MANAGES": "MANAGED_BY", "GOVERNS": "MANAGED_BY",
    "GOVERNED_BY": "MANAGED_BY", "SUPERVISED_BY": "MANAGED_BY",
    "SUPERVISED": "MANAGED_BY", "DIRECTED_BY": "MANAGED_BY",
    "LEADS": "MANAGED_BY", "LEAD_BY": "MANAGED_BY",
    "REGULATES": "REGULATED_BY",
    "TRANSFERS": "TRANSFERRED_TO", "TRANSFERS_TO": "TRANSFERRED_TO",
    "TRANSFERRED_BY": "TRANSFERRED_TO", "TRANSFERRED_VIA": "TRANSFERRED_TO",
    "SUBMITS": "SUBMITTED_TO", "SUBMITS_TO": "SUBMITTED_TO",
    "SUBMITTED_BY": "SUBMITTED_TO",
    "DELEGATES": "DELEGATED_TO", "ASSIGNS": "ASSIGNED_TO",
    "REPORTS": "REPORTED_TO", "REPORTS_TO": "REPORTED_TO",
    "REPORTED_BY": "REPORTED_TO",
    "NOTIFIES": "NOTIFIED_TO", "INFORMS": "NOTIFIED_TO",
    "PERMITS": "PERMITTED_TO", "PROHIBITS": "PROHIBITED_FROM",
    "REQUIRES": "REQUIRED_FOR", "AFFECTS": "AFFECTED_BY",
    "REPLACES": "REPLACED_BY", "AMENDS": "AMENDED_BY",
    "REPEALS": "REPEALED_BY", "REFERENCES": "REFERENCED_BY",
    "RELATED": "RELATED_TO", "CONNECTED_TO": "RELATED_TO",
    "IMPLEMENTS": "IMPLEMENTED_BY", "PERFORMS": "IMPLEMENTED_BY",
    "PERFORMED_BY": "IMPLEMENTED_BY", "MONITORED_BY": "MANAGED_BY",
    "INSPECTED_BY": "MANAGED_BY", "ASSESSED_BY": "MANAGED_BY",
    "VERIFIED_BY": "MANAGED_BY", "REVIEWED_BY": "MANAGED_BY",
    "PENALIZED_BY": "AFFECTED_BY", "EXPROPRIATED_BY": "AFFECTED_BY",
    "REVOKES": "REPEALED_BY", "REVOKED_BY": "REPEALED_BY",
    "SUSPENDED_BY": "REPEALED_BY", "TERMINATED_BY": "REPEALED_BY",
    "UPDATED_BY": "AMENDED_BY", "MODIFIED_BY": "AMENDED_BY",
    "CORRECTED_BY": "AMENDED_BY", "EXTENDED_BY": "AMENDED_BY",
}


def _normalize_relationship(raw_rel: str) -> str:
    clean_rel = raw_rel.upper().replace(" ", "_").strip()
    
    # 1. FIXED match
    if clean_rel in FIXED_NODE_RELATIONS:
        return clean_rel
        
    # 2. Xóa đồng nghĩa (Fuzzy Verb Root matching)
    parts = clean_rel.split("_")
    first_word = parts[0]
    
    for i in range(len(first_word), 3, -1):
        prefix = first_word[:i]
        if prefix in _VERB_ROOT_CANONICAL:
            return _VERB_ROOT_CANONICAL[prefix]
            
    # 3. Fallback: Nếu không phải từ đồng nghĩa đã biết, GIỮ NGUYÊN
    return clean_rel


def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


def query_relation_types(driver) -> list[dict]:
    """Lấy tất cả relationship types + số lượng, sorted desc."""
    q = """
    CALL db.relationshipTypes() YIELD relationshipType
    CALL apoc.cypher.run(
        'MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as cnt',
        {}
    ) YIELD value
    RETURN relationshipType AS rel_type, value.cnt AS count
    ORDER BY value.cnt DESC
    """
    # Fallback nếu không có APOC
    q_no_apoc = """
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC
    """
    with driver.session() as session:
        try:
            result = session.run(q)
            return [{"rel_type": r["rel_type"], "count": r["count"]} for r in result]
        except Exception:
            result = session.run(q_no_apoc)
            return [{"rel_type": r["rel_type"], "count": r["count"]} for r in result]


def analyze_verb_roots(rel_types: list[dict]) -> dict:
    """Nhóm các relation types theo verb root (phần trước _BY/_TO/_FROM/...)."""
    PREPS = ["_FROM", "_WITH", "_VIA", "_IN", "_AT", "_ON", "_FOR", "_TO", "_BY"]
    groups = defaultdict(list)
    for item in rel_types:
        rt = item["rel_type"]
        stem = rt
        for prep in PREPS:
            if rt.endswith(prep):
                stem = rt[:-len(prep)]
                break
        groups[stem].append(item)
    # Chỉ trả về nhóm có >1 variant
    return {k: v for k, v in groups.items() if len(v) > 1}


def simulate_normalization(rel_types: list[dict]) -> dict:
    """Mô phỏng normalization mới — xem bao nhiêu type được gộp về FIXED."""
    mapping = defaultdict(list)
    for item in rel_types:
        canonical = _normalize_relationship(item["rel_type"])
        mapping[canonical].append(item)
    return mapping


def print_separator(char="═", width=70):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Debug Neo4j relationship types")
    parser.add_argument("--uri",      default=os.getenv("TEST_NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7689")))
    parser.add_argument("--user",     default=os.getenv("TEST_NEO4J_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j")))
    parser.add_argument("--password", default=os.getenv("TEST_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "")))
    parser.add_argument("--top",      type=int, default=30, help="Hiển thị top N types")
    parser.add_argument("--noise-threshold", type=int, default=5,
                        help="Types có count <= threshold được coi là noise")
    args = parser.parse_args()

    print(f"\n🔌 Kết nối Neo4j: {args.uri}")
    driver = get_driver(args.uri, args.user, args.password)

    print("⏳ Truy vấn relationship types...")
    rel_types = query_relation_types(driver)
    driver.close()

    total_types  = len(rel_types)
    total_rels   = sum(r["count"] for r in rel_types)
    noise_types  = [r for r in rel_types if r["count"] <= args.noise_threshold]
    fixed_types  = [r for r in rel_types if r["rel_type"] in FIXED_NODE_RELATIONS]

    # ── 1. TỔNG QUAN ──────────────────────────────────────────────────
    print_separator()
    print(f"  📊 TỔNG QUAN RELATIONSHIP GRAPH")
    print_separator()
    print(f"  Tổng số relationship types : {total_types:,}")
    print(f"  Tổng số relationships      : {total_rels:,}")
    print(f"  Types thuộc FIXED set      : {len(fixed_types)} / {len(FIXED_NODE_RELATIONS)}")
    print(f"  Types 'noise' (≤{args.noise_threshold} rels)    : {len(noise_types):,} types")
    print()

    # ── 2. TOP N TYPES ────────────────────────────────────────────────
    print_separator("─")
    print(f"  🏆 TOP {args.top} RELATIONSHIP TYPES")
    print_separator("─")
    print(f"  {'#':<4} {'TYPE':<45} {'COUNT':>10}  {'IN_FIXED':>8}")
    print_separator("─")
    fixed_set = FIXED_NODE_RELATIONS
    for i, item in enumerate(rel_types[:args.top], 1):
        in_fixed = "✅" if item["rel_type"] in fixed_set else "❌"
        print(f"  {i:<4} {item['rel_type']:<45} {item['count']:>10,}  {in_fixed:>8}")
    print()

    # ── 3. NHÓM THEO VERB ROOT (biến thể cùng gốc) ───────────────────
    print_separator("─")
    print("  🔗 NHÓM BIẾN THỂ CÙNG GỐC ĐỘNG TỪ (có >1 variant)")
    print_separator("─")
    verb_groups = analyze_verb_roots(rel_types)
    # Sort theo tổng count giảm dần
    sorted_groups = sorted(
        verb_groups.items(),
        key=lambda x: sum(v["count"] for v in x[1]),
        reverse=True
    )
    for stem, variants in sorted_groups[:25]:
        total_count = sum(v["count"] for v in variants)
        print(f"\n  ROOT: {stem}  (tổng: {total_count:,} rels, {len(variants)} variants)")
        for v in sorted(variants, key=lambda x: x["count"], reverse=True):
            in_fixed = "✅" if v["rel_type"] in fixed_set else "❌"
            print(f"    {in_fixed}  {v['rel_type']:<45} {v['count']:>8,}")

    # ── 4. MÔ PHỎNG NORMALIZATION ─────────────────────────────────────
    print()
    print_separator()
    print("  🔄 MÔ PHỎNG NORMALIZATION MỚI")
    print_separator()
    sim = simulate_normalization(rel_types)
    types_after  = len(sim)
    types_before = total_types
    pct_reduction = (1 - types_after / max(types_before, 1)) * 100

    print(f"  Types trước normalization : {types_before:,}")
    print(f"  Types sau  normalization  : {types_after:,}")
    print(f"  Giảm                      : {types_before - types_after:,} types ({pct_reduction:.1f}%)")
    print()
    print(f"  {'CANONICAL':<35} {'#ORIGINAL_TYPES':>15}  {'TOTAL_RELS':>12}")
    print_separator("─")
    for canonical, originals in sorted(sim.items(), key=lambda x: sum(v["count"] for v in x[1]), reverse=True):
        total_in_group = sum(v["count"] for v in originals)
        print(f"  {canonical:<35} {len(originals):>15}  {total_in_group:>12,}")
        if len(originals) > 1:
            for orig in sorted(originals, key=lambda x: x["count"], reverse=True):
                print(f"    ↳ {orig['rel_type']:<33} {orig['count']:>12,}")
    print()

    # ── 5. NOISE REPORT ───────────────────────────────────────────────
    print_separator("─")
    print(f"  🗑️  NOISE TYPES (count ≤ {args.noise_threshold}) — {len(noise_types)} types")
    print_separator("─")
    for item in sorted(noise_types, key=lambda x: x["count"]):
        canonical = _normalize_relationship(item["rel_type"])
        print(f"  {item['rel_type']:<45} cnt={item['count']:<5}  →  {canonical}")

    print()
    print_separator()
    print("  ✅ Hoàn thành phân tích.")
    print_separator()


if __name__ == "__main__":
    main()
