import os
import sys
import re
from dotenv import load_dotenv

# Fix UTF-8 cho Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[ERROR] Cần cài đặt neo4j driver: pip install neo4j")
    sys.exit(1)

# =====================================================================
# DANH SÁCH RELATION TỪ CHUNKING - TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO
# =====================================================================
CHUNKING_RELATIONS = {
    "HAS_ENTITY", "HAS_ARTICLE", "HAS_TYPE", "HAS_SECTOR", 
    "PART_OF", "BELONGS_TO", "ISSUED_BY", "SIGNED_BY", "BASED_ON"
}

# =====================================================================
# NORMALIZATION CONSTANTS TỪ entities.py
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

_RE_HAS_PROP = re.compile(r'^HAS_')

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
    "REQUIR": "REQUIRED_FOR", "COMPLY": "COMPLIES_WITH", "COMPLI": "COMPLIES_WITH",
    "AFFECT": "AFFECTED_BY", "PENALIZ": "AFFECTED_BY",
    "DEFIN": "DEFINED_IN", "CLASSIFY": "CLASSIFIED_AS", "CLASSIFI": "CLASSIFIED_AS",
    "BELONG": "BELONGS_TO", "LOCAT": "LOCATED_IN",
    "FUND": "FUNDED_BY", "PAY": "PAID_TO", "PAID": "PAID_TO", "COLLECT": "COLLECTED_BY",
    "REPLAC": "REPLACED_BY", "AMEND": "AMENDED_BY", "REPEAL": "REPEALED_BY",
    "REFERENCE": "REFERENCED_BY", "BASE": "BASED_ON",
    "RELATED": "RELATED_TO",
}

def _normalize_relationship(raw_rel: str) -> str:
    clean_rel = raw_rel.upper().replace(" ", "_").strip()
    
    # 1. FIXED match (đã chuẩn thì giữ nguyên)
    if clean_rel in FIXED_NODE_RELATIONS:
        return clean_rel
        
    # 2. Xóa đồng nghĩa (Fuzzy Verb Root matching)
    parts = clean_rel.split("_")
    first_word = parts[0]
    
    # Cắt dần từ cuối để tìm root (vd: IMPLEMENTED -> IMPLEMENT)
    for i in range(len(first_word), 3, -1):
        prefix = first_word[:i]
        if prefix in _VERB_ROOT_CANONICAL:
            return _VERB_ROOT_CANONICAL[prefix]
            
    # 3. Fallback: Nếu không phải từ đồng nghĩa đã biết, GIỮ NGUYÊN
    # (Bảo tồn nghĩa gốc hoặc các relation xuất hiện ít)
    return clean_rel

def get_driver(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"[ERROR] Không thể kết nối Neo4j: {e}")
        sys.exit(1)

def query_all_relationship_types(driver):
    """Lấy danh sách các loại relationship đang có trong database."""
    q = """
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC
    """
    with driver.session() as session:
        result = session.run(q)
        return [{"rel_type": record["rel_type"], "count": record["count"]} for record in result]

def rename_relationship(driver, old_type: str, new_type: str):
    """
    Đổi tên relationship trong Neo4j.
    Chỉ thực hiện nếu old_type khác new_type.
    """
    if old_type == new_type:
        return 0
        
    q = f"""
    MATCH ()-[r:`{old_type}`]->()
    WITH r LIMIT 100000
    CREATE (startNode(r))-[new_r:`{new_type}`]->(endNode(r))
    SET new_r = properties(r)
    DELETE r
    RETURN count(new_r) as renamed
    """
    
    total_renamed = 0
    with driver.session() as session:
        while True:
            result = session.run(q)
            renamed = result.single()["renamed"]
            if renamed == 0:
                break
            total_renamed += renamed
            print(f"    ... renamed batch of {renamed} '{old_type}' -> '{new_type}'")
            
    return total_renamed

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate Neo4j relationships to canonical forms")
    parser.add_argument("--uri",      default=os.getenv("TEST_NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7689")))
    parser.add_argument("--user",     default=os.getenv("TEST_NEO4J_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j")))
    parser.add_argument("--password", default=os.getenv("TEST_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "")))
    parser.add_argument("--dry-run",  action="store_true", help="Chỉ in ra kết quả mô phỏng, KHÔNG ghi vào DB")
    args = parser.parse_args()

    print(f"\n🔌 Kết nối Neo4j: {args.uri}")
    driver = get_driver(args.uri, args.user, args.password)

    print("⏳ Phân tích các relationships hiện tại...")
    all_types = query_all_relationship_types(driver)
    
    migration_plan = []
    
    for item in all_types:
        rel_type = item["rel_type"]
        count = item["count"]
        
        # BỎ QUA CÁC RELATION CỦA CHUNKING
        if rel_type in CHUNKING_RELATIONS:
            # print(f"  [SKIP] {rel_type} (Chunking structural relation) - {count} rels")
            continue
            
        canonical = _normalize_relationship(rel_type)
        if canonical != rel_type:
            migration_plan.append({
                "old": rel_type,
                "new": canonical,
                "count": count
            })
            
    if not migration_plan:
        print("✅ Tất cả các relationships (ngoài chunking) đều đã chuẩn. Không cần migrate.")
        driver.close()
        return
        
    print(f"\n📋 KẾ HOẠCH MIGRATION ({len(migration_plan)} types cần đổi):")
    for task in migration_plan:
        print(f"  - Đổi {task['old']:<30} -> {task['new']:<20} ({task['count']} rels)")
        
    if args.dry_run:
        print("\n🛑 Đây là DRY RUN. Không có thay đổi nào được thực hiện trong DB.")
        driver.close()
        return
        
    print("\n🚀 BẮT ĐẦU MIGRATION...")
    total_migrated = 0
    for i, task in enumerate(migration_plan, 1):
        print(f"[{i}/{len(migration_plan)}] Migrating '{task['old']}' -> '{task['new']}' ({task['count']} rels)...")
        renamed_count = rename_relationship(driver, task["old"], task["new"])
        total_migrated += renamed_count
        
    print(f"\n✅ Migration hoàn tất. Tổng cộng đã rename {total_migrated} relationships.")
    driver.close()

if __name__ == "__main__":
    main()
