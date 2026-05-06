from typing import Dict, List
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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from backend.config import (
    FIXED_NODE_RELATIONS, BLACKLIST_RELATIONS, FIXED_DOC_RELATIONS,
    _VERB_ROOT_CANONICAL, _CROSS_VERB_MAPPING
)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[ERROR] neo4j driver chua cai: pip install neo4j")
    sys.exit(1)

# =====================================================================
# Inline normalization constants (tranh import entities.py vi no keo
# theo icllmlib / LLM client khong can thiet cho debug script nay)
# =====================================================================


# Regex: bắt toàn bộ HAS_* property giả



# Tập động: ghi nhận nhãn mới LLM tạo trong quá trình pipeline
# → các lần normalize sau sẽ khoanh ngắ luôn mà không force fallback
def _normalize_relationship(raw_rel: str) -> str:
    """
    Pipeline chuẩn hoá quan hệ — CLOSED SET, không tạo DYNAMIC mới:
    0. HAS_* regex → RELATED_TO
    1. Blacklist → RELATED_TO
    2. Comprehensive alias_map (chủ động→bị động, đồng nghĩa→canonical)
    3. Exact match FIXED
    4. Verb-root fuzzy match → FIXED canonical
    5. Fallback RELATED_TO
    """
    if not raw_rel:
        return "RELATED_TO"
    s = raw_rel.strip().upper().replace(" ", "_")

    # 0. Ánh xạ cận nghĩa / dọn rác
    if s in _CROSS_VERB_MAPPING:
        s = _CROSS_VERB_MAPPING[s]


    # 1. Blacklist
    if s in BLACKLIST_RELATIONS:
        return "RELATED_TO"

    # 2. Comprehensive alias map
    _ALIAS: dict = {
        # --- Chủ động → Bị động (Issuance / Acceptance) ---
        "ISSUES":              "ISSUED_BY",
        "ISSUED":              "ISSUED_BY",
        "SIGNS":               "SIGNED_BY",
        "SIGNED":              "SIGNED_BY",
        "APPROVES":            "APPROVED_BY",
        "APPROVED":            "APPROVED_BY",
        "PUBLISHES":           "PUBLISHED_BY",
        "PUBLISHED":           "PUBLISHED_BY",
        "PUBLISHES_IN":        "PUBLISHED_BY",
        "CREATES":             "CREATED_BY",
        "CREATED":             "CREATED_BY",
        "ESTABLISHES":         "ESTABLISHED_BY",
        "ESTABLISHED":         "ESTABLISHED_BY",
        "ACCEPTS":             "ACCEPTED_BY",
        "CONFIRMS":            "CONFIRMED_BY",
        "ACKNOWLEDGES":        "ACKNOWLEDGED_BY",
        # --- Implementation ---
        "IMPLEMENTS":          "IMPLEMENTED_BY",
        "PERFORMS":            "IMPLEMENTED_BY",
        "PERFORMED_BY":        "IMPLEMENTED_BY",
        "CARRIED_OUT_BY":      "IMPLEMENTED_BY",
        "DIRECTS_IMPLEMENTATION_OF": "IMPLEMENTED_BY",
        "GUIDES_IMPLEMENTATION_OF":  "IMPLEMENTED_BY",
        "ENFORCES":            "ENFORCED_BY",
        "APPLIES":             "APPLIED_BY",
        "EXECUTES":            "EXECUTED_BY",
        "EXECUTED":            "EXECUTED_BY",
        "EXECUTES_AT":         "EXECUTED_BY",
        # --- Management synonyms → MANAGED_BY ---
        "MANAGES":             "MANAGED_BY",
        "GOVERNS":             "MANAGED_BY",
        "GOVERNED_BY":         "MANAGED_BY",
        "SUPERVISED_BY":       "MANAGED_BY",
        "SUPERVISES":          "MANAGED_BY",
        "DIRECTED_BY":         "MANAGED_BY",
        "DIRECTS":             "MANAGED_BY",
        "LEADS":               "MANAGED_BY",
        "LEAD_BY":             "MANAGED_BY",
        "OPERATES_UNDER":      "MANAGED_BY",
        "SUBORDINATE_TO":      "MANAGED_BY",
        "IS_RESPONSIBLE_FOR":  "MANAGED_BY",
        "RESPONSIBLE_FOR":     "MANAGED_BY",
        "ACCOUNTABLE_TO":      "MANAGED_BY",
        "IS_HIGHEST_AUTHORITY_OF": "MANAGED_BY",
        "CHAIRMANED_BY":       "MANAGED_BY",
        "ASSESSES":            "MANAGED_BY",
        "DECIDES":             "DECIDED_BY",
        "DETERMINES":          "DETERMINED_BY",
        "ORGANIZES":           "ORGANIZED_BY",
        "INITIATES":           "INITIATED_BY",
        "GUIDES":              "GUIDED_BY",
        "HANDLES":             "HANDLED_BY",
        # --- Regulation ---
        "REGULATES":           "REGULATED_BY",
        "COORDINATES":         "COORDINATED_BY",
        "COORDINATED_WITH":    "COORDINATED_BY",
        "COORDINATES_WITH":    "COORDINATED_BY",
        "CONTRACTS_WITH":      "COOPERATES_WITH",
        "ORGANIZES_WITH":      "ORGANIZED_WITH",
        "PARTICIPATES_IN":     "PARTICIPATED_BY",
        "CONSULTED_WITH":      "CONSULTED_BY",
        # --- Transfer / Assignment ---
        "TRANSFERS":           "TRANSFERRED_TO",
        "TRANSFERS_TO":        "TRANSFERRED_TO",
        "TRANSFERRED_BY":      "TRANSFERRED_TO",
        "TRANSFERRED_VIA":     "TRANSFERRED_TO",
        "TRANSPORTS":          "TRANSFERRED_TO",
        "DELIVERS":            "TRANSFERRED_TO",
        "DELIVERED_TO":        "TRANSFERRED_TO",
        "SUBMITS":             "SUBMITTED_TO",
        "SUBMITS_TO":          "SUBMITTED_TO",
        "SUBMITTED_BY":        "SUBMITTED_TO",
        "DELEGATES":           "DELEGATED_TO",
        "DELEGATES_TO":        "DELEGATED_TO",
        "ASSIGNS":             "ASSIGNED_TO",
        "ASSIGNED":            "ASSIGNED_TO",
        "RETURNED_TO":         "TRANSFERRED_TO",
        "RETURNED_BY":         "TRANSFERRED_TO",
        "APPOINTS":            "APPOINTED_BY",
        "EMPLOYS":             "EMPLOYED_BY",
        "IMPORTS":             "IMPORTED_BY",
        "PROPOSED_TO":         "SUBMITTED_TO",
        # --- Reporting / Notification ---
        "REPORTS":             "REPORTED_TO",
        "REPORTS_TO":          "REPORTED_TO",
        "REPORTED_BY":         "REPORTED_TO",
        "REPORTS_ON":          "REPORTED_TO",
        "REPORTS_VIOLATIONS_TO": "REPORTED_TO",
        "NOTIFIES":            "NOTIFIED_TO",
        "NOTIFIED_BY":         "NOTIFIED_TO",
        "INFORMS":             "NOTIFIED_TO",
        "PROVIDES_INFO_TO":    "NOTIFIED_TO",
        "PUBLISHES_INFO_TO":   "NOTIFIED_TO",
        "TRANSMITS_DATA_TO":   "NOTIFIED_TO",
        "REQUESTS_INFO_FROM":  "NOTIFIED_TO",
        "RECEIVES_INFO_FROM":  "NOTIFIED_TO",
        "RECEIVES":            "RECEIVED_BY",
        "RECEIVED_FROM":       "RECEIVED_BY",
        # --- Permission / Prohibition ---
        "PERMITS":             "PERMITTED_TO",
        "PERMITTED_FOR":       "PERMITTED_TO",
        "PERMITTED_TO_USE":    "PERMITTED_TO",
        "CAN_PERFORM":         "PERMITTED_TO",
        "PROHIBITS":           "PROHIBITED_FROM",
        "PROHIBITED_FOR":      "PROHIBITED_FROM",
        "PROHIBITED_IN":       "PROHIBITED_FROM",
        "PROHIBITED_BY":       "PROHIBITED_FROM",
        "PREVENTED_BY":        "PROHIBITED_FROM",
        "PROHIBITED_FROM_CONTAINING": "PROHIBITED_FROM",
        "MUST_NOT_PERFORM":    "PROHIBITED_FROM",
        "MUST_NOT_MISLEAD":    "PROHIBITED_FROM",
        "MUST_PROTECT":        "PROTECTED_BY",
        "PROTECTS":            "PROTECTED_BY",
        "EXEMPT":              "EXEMPT_FROM",
        "EXCLUDED_FROM":       "EXEMPT_FROM",
        "EXEMPTS":             "EXEMPT_FROM",
        "ENTITLES":            "ENTITLED_TO",
        "PRIORITY_FOR":        "ENTITLED_TO",
        "PRIORITY_OVER":       "ENTITLED_TO",
        "HAS_PRIORITY_ACCESS_TO": "ENTITLED_TO",
        # --- Requirement / Compliance ---
        "REQUIRES":            "REQUIRED_FOR",
        "REQUIRED":            "REQUIRED_FOR",
        "MUST_COMPLY_WITH":    "COMPLIES_WITH",
        "MUST_MEET":           "COMPLIES_WITH",
        "CONFORMS_TO":         "COMPLIES_WITH",
        "CONDITIONED_ON":      "COMPLIES_WITH",
        "ENSURES_COMPLIANCE_WITH": "COMPLIES_WITH",
        "COMPLIES":            "COMPLIES_WITH",
        "ENSURES":             "ENSURED_BY",
        "AFFECTED":            "AFFECTED_BY",
        "AFFECTS":             "AFFECTED_BY",
        "SUBJECTED_TO":        "AFFECTED_BY",
        # --- Definition / Classification ---
        "DEFINES":             "DEFINED_IN",
        "DEFINED_AS":          "DEFINED_IN",
        "DEFINED_BY":          "DEFINED_IN",
        "DEFINED_FOR":         "DEFINED_IN",
        "DEFINED_BY_ABSENCE_OF": "DEFINED_IN",
        "CLASSIFIED_IN":       "CLASSIFIED_AS",
        "CLASSIFIED":          "CLASSIFIED_AS",
        "MARKED_WITH":         "CLASSIFIED_AS",
        "NAMED_AFTER":         "CLASSIFIED_AS",
        "IS_LEGAL_REPRESENTATIVE_OF": "BELONGS_TO",
        "CONTAINED_IN":        "PART_OF",
        "CONTAINS":            "PART_OF",
        "INCLUDED_IN":         "PART_OF",
        "INCLUDES":            "PART_OF",
        "ATTACHED_TO":         "PART_OF",
        "WORKS_FOR":           "BELONGS_TO",
        "WORKS_IN":            "BELONGS_TO",
        "REPRESENTS":          "REPRESENTED_BY",
        # --- Financial ---
        "FUNDS":               "FUNDED_BY",
        "PAYS":                "PAID_TO",
        "PAID_FOR":            "PAID_TO",
        "DEDUCTED_FROM":       "PAID_TO",
        "COMPENSATED_BY":      "PAID_BY",
        "COLLECTS":            "COLLECTED_BY",
        "DEPOSITED_TO":        "PAID_TO",
        "PURCHASED_FROM":      "PAID_BY",
        # --- Financial Synonyms ---
        "FUNDED":              "FUNDED_BY",
        "ALLOCATED_BY":        "FUNDED_BY",
        "ALLOCATED_FOR":       "FUNDED_BY",
        "ALLOCATED_FROM":      "FUNDED_BY",
        "ALLOCATED_TO":        "FUNDED_BY",
        "ALLOCATED_VIA":       "FUNDED_BY",
        "ALLOCATES":           "FUNDED_BY",
        # --- Document synonyms ---
        "REPLACES":            "REPLACED_BY",
        "AMENDS":              "AMENDED_BY",
        "AMENDED":             "AMENDED_BY",
        "REPEALS":             "REPEALED_BY",
        "REMOVED_FROM":        "REPEALED_BY",
        "REFERENCES":          "REFERENCED_BY",
        "REFERS_TO":           "REFERENCED_BY",
        "REFERRED_BY":         "REFERENCED_BY",
        # --- General ---
        "ISSUED_FOR":          "APPLIES_TO",
        "DESIGNED_FOR":        "APPLIES_TO",
        "ORGANIZED_FOR":       "APPLIES_TO",
        "PERMITTED_BY":        "APPLIES_TO",
        "APPLIES":             "APPLIES_TO",
        "RELATED":             "RELATED_TO",
        "CONNECTED_TO":        "RELATED_TO",
        "SYNCHRONIZED_WITH":   "RELATED_TO",
        "IS_EQUAL_TO":         "RELATED_TO",
        "SUPPLEMENTED_BY":     "RELATED_TO",
        "COVERS":              "RELATED_TO",
        "COVERED_BY":          "RELATED_TO",
        "LEASED_BY":           "RELATED_TO",
        "ASSUMES":             "RELATED_TO",
        "SEPARATED_FROM":      "RELATED_TO",
        "STORED_AT":           "LOCATED_IN",
        # --- Other active → canonical ---
        "PRODUCES":            "CREATED_BY",
        "GENERATED_BY":        "CREATED_BY",
        "GENERATES":           "CREATED_BY",
        "REVOKES":             "REPEALED_BY",
        "REVOKED_BY":          "REPEALED_BY",
        "SUSPENDED_BY":        "REPEALED_BY",
        "TERMINATED_BY":       "REPEALED_BY",
        "UPDATED_BY":          "AMENDED_BY",
        "UPDATES":             "AMENDED_BY",
        "MODIFIED_BY":         "AMENDED_BY",
        "MODIFIED_VIA":        "AMENDED_BY",
        "CORRECTED_BY":        "AMENDED_BY",
        "EXTENDED_BY":         "AMENDED_BY",
        "SUPPORTED_BY":        "IMPLEMENTED_BY",
        "SUPPORTS":            "IMPLEMENTED_BY",
        "ASSISTED_BY":         "IMPLEMENTED_BY",
        "MAINTAINED_BY":       "IMPLEMENTED_BY",
        "MAINTAINS":           "IMPLEMENTED_BY",
        "SERVICED_BY":         "IMPLEMENTED_BY",
        "SERVICES":            "IMPLEMENTED_BY",
        "PROCESSED_BY":        "IMPLEMENTED_BY",
        "PROCESSED_IN":        "IMPLEMENTED_BY",
        "TEACHES":             "EDUCATED_BY",
        "TRAINING_PROVIDED_BY": "EDUCATED_BY",
        "TRIGGERS":            "TRIGGERED_BY",
        "USES":                "USED_BY",
        "PROVIDES":            "PROVIDED_BY",
        "PROVIDES_TO":         "PROVIDED_TO",
        "OPENS":               "OPENED_BY",
        "DEVELOPS":            "DEVELOPED_BY",
        "PREPARES":            "PREPARED_BY",
        "PREPARED_WITH":       "PREPARED_BY",
        "AWARDS":              "AWARDED_BY",
        "AWARDED_WITH":        "AWARDED_TO",
        "CALCULATED_FROM":     "CALCULATED_BY",
        "MONITORED_BY":        "MANAGED_BY",
        "INSPECTED_BY":        "MANAGED_BY",
        "ASSESSED_BY":         "MANAGED_BY",
        "EVALUATED_BY":        "MANAGED_BY",
        "VERIFIED_BY":         "MANAGED_BY",
        "REVIEWED_BY":         "MANAGED_BY",
        "REVIEWS":             "MANAGED_BY",
        "ANALYZED_BY":         "MANAGED_BY",
        "AUDITED_BY":          "MANAGED_BY",
        "INVESTIGATED_BY":     "MANAGED_BY",
        "PENALIZED_BY":        "AFFECTED_BY",
        "CONFISCATED_BY":      "AFFECTED_BY",
        "DETAINED_BY":         "AFFECTED_BY",
        "EXPROPRIATED_BY":     "AFFECTED_BY",
        "EXPROPRIATES":        "AFFECTED_BY",
        "PROPOSES":            "PROPOSED_BY",
        "ELECTS":              "ELECTED_BY",
    }
    if s in _ALIAS:
        s = _ALIAS[s]

    # 3. Exact match FIXED (sau alias)
    if s in FIXED_NODE_RELATIONS:
        return s

    # 4. Verb-root fuzzy match → canonical FIXED
    first_word = s.split('_')[0]
    for i in range(len(first_word), 3, -1):
        prefix = first_word[:i]
        if prefix in _VERB_ROOT_CANONICAL:
            return _VERB_ROOT_CANONICAL[prefix]

    # 5. Fallback nếu không khớp bất cứ quy tắc nào -> GIỮ NGUYÊN (bảo tồn)
    return s




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
