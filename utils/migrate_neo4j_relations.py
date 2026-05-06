from typing import Dict, List
import os
import sys
import re
from dotenv import load_dotenv

# Fix UTF-8 cho Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from backend.config import (
    CHUNKING_RELATIONS, FIXED_NODE_RELATIONS, BLACKLIST_RELATIONS,
    FIXED_DOC_RELATIONS, _VERB_ROOT_CANONICAL, _CROSS_VERB_MAPPING
)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[ERROR] Cần cài đặt neo4j driver: pip install neo4j")
    sys.exit(1)









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
    Đổi tên relationship trong Neo4j dùng batched loop.
    - Escape backtick trong tên relation.
    - Mỗi batch chạy trong session riêng để tránh timeout.
    - Guard None trên result.single().
    """
    if old_type == new_type:
        return 0

    # Escape backtick trong tên relation (tránh Cypher injection)
    safe_old = old_type.replace("`", "``")
    safe_new = new_type.replace("`", "``")
    batch_size = 10_000  # Nhỏ hơn để tránh timeout / heap OOM

    total_renamed = 0
    while True:
        q = f"""
        MATCH (a)-[r:`{safe_old}`]->(b)
        WITH a, r, b LIMIT {batch_size}
        CREATE (a)-[new_r:`{safe_new}`]->(b)
        SET new_r = properties(r)
        DELETE r
        RETURN count(*) AS renamed
        """
        with driver.session() as session:
            try:
                record = session.run(q).single()
                renamed = record["renamed"] if record else 0
            except Exception as e:
                print(f"    [ERROR] Batch failed for '{old_type}': {e}")
                raise

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
