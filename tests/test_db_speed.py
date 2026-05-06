import os
import sys
import time
import uuid
import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Tải biến môi trường
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

try:
    load_dotenv(os.path.join(repo_root, ".env"), override=False)
except ImportError:
    pass

# Đọc cấu hình kết nối từ .env (production)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6337")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_rag_docs_nam")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "")

# Output file
RESULT_FILE = os.path.join(repo_root, "test_results.txt")
_result_lines = []  # buffer ket qua de ghi file cuoi cung

def _out(msg: str):
    """In ra terminal + buffer de ghi file."""
    print(msg)
    _result_lines.append(msg)

def _save_results():
    """Ghi buffer ra file test_results.txt (append)."""
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"test_db_speed.py — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        for line in _result_lines:
            f.write(line + "\n")

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def measure_time(func):
    """Decorator để đo thời gian thực thi"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        _out(f"  Thoi gian: {(end_time - start_time) * 1000:.2f} ms")
        return result
    return wrapper

# ==============================================================================
# KỊCH BẢN TEST QDRANT (VECTOR SEARCH) - TỪ DỄ ĐẾN KHÓ
# ==============================================================================
@measure_time
def test_qdrant_easy_semantic_search(client, collection_name, query_vector=None):
    print("\n[QDRANT - DỄ] 🔍 Scenario 1: Semantic Search Cơ Bản")
    print("Mô tả: Tìm kiếm các chunk có ý nghĩa tương đồng với một vector (top 5).")
    if query_vector is None:
        query_vector = [0.01] * 768 
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )
        print(f"✅ Tìm thấy {len(results)} kết quả.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

@measure_time
def test_qdrant_medium_filtered_search(client, collection_name, query_vector=None, doc_id="38/2020/QĐ-UBND"):
    print(f"\n[QDRANT - TRUNG BÌNH] 🔍 Scenario 2: Semantic Search kèm Metadata Filter (doc_id = {doc_id})")
    print("Mô tả: Tìm kiếm ngữ nghĩa nhưng giới hạn bắt buộc trong một văn bản cụ thể.")
    if query_vector is None:
        query_vector = [0.01] * 768
        
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=5
        )
        print(f"✅ Tìm thấy {len(results)} kết quả.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

# ==============================================================================
# KỊCH BẢN TEST NEO4J (GRAPH TRAVERSAL) - TỪ DỄ ĐẾN KHÓ
# ==============================================================================

# ----------------- CẤP ĐỘ DỄ (EASY) -----------------
@measure_time
def test_neo4j_easy_node_lookup(driver, node_label="Organization", name_contains="Y tế"):
    print(f"\n[NEO4J - DỄ] 🕸️ Scenario 1: Tra cứu Node (Lookup)")
    print(f"Mô tả: Tìm trực tiếp các Node có label là '{node_label}' và tên chứa '{name_contains}'.")
    query = f"""
    MATCH (n:{node_label})
    WHERE toLower(n.name) CONTAINS toLower($name_contains)
    RETURN n.name AS name
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, name_contains=name_contains)
        records = list(result)
        print(f"✅ Tìm thấy {len(records)} node.")

@measure_time
def test_neo4j_easy_1_hop(driver, org_name="Bộ Y tế"):
    print(f"\n[NEO4J - DỄ] 🕸️ Scenario 2: Traversal 1 Bước (1-hop) - Quan hệ cơ bản")
    print(f"Mô tả: Tìm tất cả Document có quan hệ ISSUED_BY với Organization '{org_name}'.")
    query = """
    MATCH (d:Document)-[:ISSUED_BY]->(o:Organization)
    WHERE toLower(o.name) CONTAINS toLower($org_name)
    RETURN d.document_number AS doc_num, o.name AS org
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, org_name=org_name)
        records = list(result)
        print(f"✅ Tìm thấy {len(records)} văn bản.")

# ----------------- CẤP ĐỘ TRUNG BÌNH (MEDIUM) -----------------
@measure_time
def test_neo4j_medium_doc_hierarchy(driver, doc_number="38/2020/QĐ-UBND"):
    print(f"\n[NEO4J - TRUNG BÌNH] 🕸️ Scenario 3: Traversal Phân cấp Văn bản (Document -> Article -> Clause -> Chunk)")
    print(f"Mô tả: Lấy toàn bộ cấu trúc cây của một văn bản (Document) xuống tới Chunk.")
    query = """
    MATCH (d:Document)-[:HAS_ARTICLE|PART_OF*1..3]->(child)
    WHERE d.document_number = $doc_number
    RETURN labels(child)[0] AS type, count(child) AS count
    """
    with driver.session() as session:
        result = session.run(query, doc_number=doc_number)
        records = list(result)
        print(f"✅ Phân tách cấu trúc văn bản: {records}")

@measure_time
def test_neo4j_medium_all_1_hop_radius(driver, concept_name="An toàn thực phẩm"):
    print(f"\n[NEO4J - TRUNG BÌNH] 🕸️ Scenario 4: Bán kính 1 Bước (1-hop Radius) từ một Concept")
    print(f"Mô tả: Lấy *tất cả* các Node và loại quan hệ có kết nối trực tiếp (vào hoặc ra) với Concept '{concept_name}'.")
    query = """
    MATCH (c:Concept)-[r]-(connected_node)
    WHERE toLower(c.name) CONTAINS toLower($concept_name)
    RETURN type(r) AS rel_type, labels(connected_node)[0] AS node_type, count(connected_node) AS freq
    ORDER BY freq DESC
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, concept_name=concept_name)
        records = list(result)
        print(f"✅ Thống kê bán kính 1 bước quanh '{concept_name}':")
        for r in records[:5]:
            print(f"   - {r['rel_type']} -> {r['node_type']} (Số lượng: {r['freq']})")

# ----------------- CẤP ĐỘ KHÓ (HARD) -----------------
@measure_time
def test_neo4j_hard_2_hop_radius(driver, org_name="Bộ Y tế"):
    print(f"\n[NEO4J - KHÓ] 🕸️ Scenario 5: Bán kính 2 Bước (2-hop Radius) - Góc nhìn bao quát")
    print(f"Mô tả: Tìm *tất cả* mọi thứ cách '{org_name}' đúng 2 bước nhảy (Organization -> Document -> Entities).")
    # Ví dụ: Bộ Y tế -> ISSUED -> Document -> HAS_ENTITY -> Procedure
    query = """
    MATCH (o:Organization)-[r1]-(mid)-[r2]-(target)
    WHERE toLower(o.name) CONTAINS toLower($org_name)
      AND NOT target:Organization  // Bỏ qua việc trỏ lại chính tổ chức khác
    RETURN labels(target)[0] AS target_type, type(r2) AS rel2, type(r1) AS rel1, count(target) AS freq
    ORDER BY freq DESC
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, org_name=org_name)
        records = list(result)
        print(f"✅ Thống kê bán kính 2 bước từ '{org_name}':")
        for r in records[:5]:
            print(f"   - [Org] -{r['rel1']}- [Mid] -{r['rel2']}- [{r['target_type']}] (Số lượng: {r['freq']})")

@measure_time
def test_neo4j_hard_cross_doc_links(driver):
    print("\n[NEO4J - KHÓ] 🕸️ Scenario 6: Liên kết chéo giữa các Văn bản (Cross-document Traversal)")
    print("Mô tả: Tìm các văn bản (D1) sửa đổi/căn cứ vào văn bản (D2) VÀ tìm các Tổ chức phát hành cả hai văn bản đó.")
    query = """
    MATCH (d1:Document)-[r:AMENDED_BY|BASED_ON|REPEALED_BY|REPLACED_BY]->(d2:Document)
    MATCH (d1)-[:ISSUED_BY]->(org1:Organization)
    MATCH (d2)-[:ISSUED_BY]->(org2:Organization)
    RETURN d1.document_number AS doc1, org1.name AS o1, type(r) AS relation, d2.document_number AS doc2, org2.name AS o2
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query)
        records = list(result)
        print(f"✅ Tìm thấy {len(records)} liên kết chéo phức tạp.")
        for r in records[:3]:
            print(f"   - {r['doc1']} ({r['o1']}) -[{r['relation']}]-> {r['doc2']} ({r['o2']})")

# ----------------- CẤP ĐỘ CỰC KHÓ (VERY HARD) -----------------
@measure_time
def test_neo4j_very_hard_shortest_path(driver, concept1="An toàn thực phẩm", concept2="Y tế"):
    print(f"\n[NEO4J - CỰC KHÓ] 🕸️ Scenario 7: Đường đi ngắn nhất (Shortest Path)")
    print(f"Mô tả: Tìm đường đi ngắn nhất giữa hai Concept '{concept1}' và '{concept2}' trong toàn bộ đồ thị tri thức.")
    query = """
    MATCH (start:Concept), (end:Concept)
    WHERE toLower(start.name) CONTAINS toLower($concept1) 
      AND toLower(end.name) CONTAINS toLower($concept2)
    MATCH path = shortestPath((start)-[*]-(end))
    RETURN length(path) AS path_length, 
           [n IN nodes(path) | coalesce(n.name, n.document_number, labels(n)[0])] AS node_names
    ORDER BY path_length ASC
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, concept1=concept1, concept2=concept2)
        records = list(result)
        if records:
            print(f"✅ Tìm thấy đường đi ngắn nhất (độ dài {records[0]['path_length']}):")
            print(f"   Đường đi: {' -> '.join(records[0]['node_names'])}")
        else:
            print("❌ Không tìm thấy đường đi nào nối 2 concept này.")

@measure_time
def test_neo4j_very_hard_deep_impact_analysis(driver, condition_name="An toàn thực phẩm"):
    print(f"\n[NEO4J - CỰC KHÓ] 🕸️ Scenario 8: Phân tích Tác động Sâu (Deep Impact Analysis)")
    print(f"Mô tả: Nếu Concept/Condition '{condition_name}' thay đổi, những văn bản nào, thủ tục nào và tổ chức nào sẽ bị ảnh hưởng (bán kính lên tới 3 bước)?")
    query = """
    MATCH (c:Concept)
    WHERE toLower(c.name) CONTAINS toLower($condition_name)
    MATCH path = (c)-[*1..3]-(impacted)
    WHERE impacted:Document OR impacted:Procedure OR impacted:Organization
    RETURN labels(impacted)[0] AS type, count(DISTINCT impacted) AS impact_count
    """
    with driver.session() as session:
        result = session.run(query, condition_name=condition_name)
        records = list(result)
        print(f"✅ Tác động của việc thay đổi '{condition_name}':")
        for r in records:
            print(f"   - Ảnh hưởng tới {r['impact_count']} {r['type']}")

if __name__ == "__main__":
    _out("==================================================")
    _out("BAT DAU KIEM TRA TOC DO VA TRUY VAN CO SO DU LIEU")
    _out("==================================================")
    _out(f"Qdrant URL: {QDRANT_URL} | Collection: {QDRANT_COLLECTION}")
    _out(f"Neo4j URI: {NEO4J_URI} | User: {NEO4J_USER}")
    
    # 1. Test Qdrant
    _out("\n" + "="*50)
    _out("TEST QDRANT")
    _out("="*50)
    try:
        qdrant_client = get_qdrant_client()
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION in collection_names:
            _out(f"[OK] Da ket noi Qdrant. Collection '{QDRANT_COLLECTION}' ton tai.")
            test_qdrant_easy_semantic_search(qdrant_client, QDRANT_COLLECTION)
            test_qdrant_medium_filtered_search(qdrant_client, QDRANT_COLLECTION)
        else:
            _out(f"[SKIP] Collection '{QDRANT_COLLECTION}' khong ton tai.")
    except Exception as e:
        _out(f"[ERROR] Ket noi Qdrant: {e}")

    # 2. Test Neo4j
    _out("\n" + "="*50)
    _out("TEST NEO4J")
    _out("="*50)
    neo4j_driver = get_neo4j_driver()
    if neo4j_driver:
        try:
            neo4j_driver.verify_connectivity()
            _out("[OK] Da ket noi Neo4j thanh cong.")
            
            # Chạy các kịch bản từ dễ đến khó
            test_neo4j_easy_node_lookup(neo4j_driver)
            test_neo4j_easy_1_hop(neo4j_driver)
            
            test_neo4j_medium_doc_hierarchy(neo4j_driver)
            test_neo4j_medium_all_1_hop_radius(neo4j_driver)
            
            test_neo4j_hard_2_hop_radius(neo4j_driver)
            test_neo4j_hard_cross_doc_links(neo4j_driver)
            
            test_neo4j_very_hard_shortest_path(neo4j_driver)
            test_neo4j_very_hard_deep_impact_analysis(neo4j_driver)
            
        except Exception as e:
            _out(f"[ERROR] Ket noi Neo4j: {e}")
        finally:
            neo4j_driver.close()
    else:
        _out("[ERROR] Khong the khoi tao Neo4j Driver.")
    
    _out("\n==================================================")
    _out("HOAN THANH KIEM TRA")
    _out("==================================================")

    # Ghi ket qua ra file
    _save_results()
    _out(f"\nKet qua da luu vao: {RESULT_FILE}")
