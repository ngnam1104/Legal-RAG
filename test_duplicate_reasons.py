import os
from neo4j import GraphDatabase

def get_driver():
    env_vars = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip()

    uri = "bolt://10.9.2.57:7688"
    user = "neo4j"
    password = "u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0"
    
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"Connected successfully to {uri}")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
    
    if driver is None:
        print("Failed to connect with all tried passwords.")
    return driver

def run_query(driver, query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]

def main():
    driver = get_driver()
    if not driver:
        print("Please ensure NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD are set correctly.")
        return

    print("="*60)
    print("PHÂN TÍCH NGUYÊN NHÂN TRÙNG LẶP DOCUMENT TRONG NEO4J")
    print("="*60)

    # 1. Test case cụ thể do user cung cấp: 1197/QĐ-UBND
    # 1. Test case cụ thể do user cung cấp: 32/QĐ-UBND
    print("\n[1] TEST CASE CHI TIẾT: 32/QĐ-UBND")
    q_specific = """
    MATCH (d:Document)
    WHERE d.document_number = '32/QĐ-UBND'
    OPTIONAL MATCH (d)-[r]->(out)
    OPTIONAL MATCH (in)-[r_in]->(d)
    RETURN properties(d) AS props, 
           count(DISTINCT r) AS num_outgoing_relations,
           count(DISTINCT r_in) AS num_incoming_relations,
           collect(DISTINCT type(r)) AS outgoing_types,
           collect(DISTINCT type(r_in)) AS incoming_types
    """
    records = run_query(driver, q_specific)
    print(f"Tìm thấy {len(records)} nodes chính xác cho '32/QĐ-UBND':")
    for idx, r in enumerate(records, 1):
        props = r['props']
        print(f"\n--- Node {idx} ---")
        print(f" - ID thật sự (props.id): {props.get('id')}")
        print(f" - is_full_text: {props.get('is_full_text')}")
        print(f" - Title: {str(props.get('title'))[:80]}...")
        print(f" - Các thuộc tính khác: { {k:v for k,v in props.items() if k not in ['id', 'title', 'is_full_text', 'document_toc']} }")
        print(f" - Tổng liên kết đi ra (outgoing): {r['num_outgoing_relations']} liên kết {r['outgoing_types']}")
        print(f" - Tổng liên kết đi vào (incoming): {r['num_incoming_relations']} liên kết {r['incoming_types']}")
    

    # 2. Phân tích tổng quan các node trùng lặp
    print("\n[2] TỔNG QUAN CÁC TRƯỜNG HỢP TRÙNG LẶP DOCUMENT_NUMBER (Top 5)")
    q_duplicates = """
    MATCH (d:Document)
    WHERE d.document_number IS NOT NULL AND d.document_number <> 'N/A' AND d.document_number <> 'unknown'
    WITH d.document_number AS doc_num, collect(d) AS nodes, count(d) AS frequency
    WHERE frequency > 1
    RETURN doc_num, frequency, 
           [n IN nodes | {id: n.id, is_full_text: n.is_full_text, year: n.year, title: coalesce(n.title, '')[0..30]}] AS node_details
    ORDER BY frequency DESC
    LIMIT 5
    """
    duplicates = run_query(driver, q_duplicates)
    for r in duplicates:
        print(f"\nDocument Number: {r['doc_num']} (Xuất hiện {r['frequency']} lần)")
        for detail in r['node_details']:
            print(f"   -> id: {detail['id']}, year: {detail['year']}, is_full_text: {detail['is_full_text']}, title: {str(detail['title'])}...")

    # 3. Phân loại nguyên nhân trùng lặp
    print("\n[3] PHÂN LOẠI CÁC NGUYÊN NHÂN TRÙNG LẶP CHÍNH")
    
    # Nguyên nhân 1: Xung đột giữa MERGE(id) cho full-text và MERGE(document_number) cho reference
    q_reason_1 = """
    MATCH (d:Document)
    WITH d.document_number AS doc_num, collect(d.id) AS ids, count(d) AS freq
    WHERE freq > 1
    WITH doc_num, ids, 
         any(i IN ids WHERE i STARTS WITH 'REF_') AS has_ref,
         any(i IN ids WHERE NOT i STARTS WITH 'REF_') AS has_full
    WHERE has_ref AND has_full
    RETURN count(doc_num) as ref_full_conflict_count
    """
    res1 = run_query(driver, q_reason_1)
    if res1:
        print(f" a) Số lượng doc_number bị trùng do 1 node là 'REF_...' và 1 node là ID thật (Lỗi logic MERGE): {res1[0]['ref_full_conflict_count']}")

    # Nguyên nhân 2: Trùng doc_number nhưng khác ID thật (Cùng số hiệu nhưng khác cơ quan/năm ban hành)
    q_reason_2 = """
    MATCH (d:Document)
    WHERE NOT d.id STARTS WITH 'REF_'
    WITH d.document_number AS doc_num, collect(d.id) AS ids, count(d) AS freq
    WHERE freq > 1
    RETURN count(doc_num) as different_docs_same_num_count
    """
    res2 = run_query(driver, q_reason_2)
    if res2:
        print(f" b) Số lượng doc_number bị trùng do các văn bản hoàn toàn khác nhau (Khác UUID, cùng số hiệu): {res2[0]['different_docs_same_num_count']}")

    # Nguyên nhân 3: Khoảng trắng thừa (Whitespace issues)
    q_reason_3 = """
    MATCH (d:Document)
    WHERE d.document_number <> trim(d.document_number)
    RETURN count(d) as whitespace_issues
    """
    res3 = run_query(driver, q_reason_3)
    if res3:
        print(f" c) Số lượng node có document_number chứa khoảng trắng thừa (ví dụ '123/QĐ '): {res3[0]['whitespace_issues']}")

    # Nguyên nhân 4: Hoa thường khác biệt (Case sensitive issues)
    q_reason_4 = """
    MATCH (d:Document)
    WHERE d.document_number IS NOT NULL AND d.document_number <> 'N/A'
    WITH toLower(d.document_number) AS lower_num, collect(d.document_number) AS variants
    WHERE size(variants) > 1 AND any(v1 IN variants WHERE any(v2 IN variants WHERE v1 <> v2))
    RETURN count(lower_num) as case_issues
    """
    res4 = run_query(driver, q_reason_4)
    if res4:
        print(f" d) Số lượng doc_number trùng nhau do viết hoa/thường (ví dụ '123/QĐ-UBND' vs '123/qđ-ubnd'): {res4[0]['case_issues']}")

    print("\n[ KẾT LUẬN ]")
    print("Dựa trên source code neo4j_client.py, có 4 nguyên nhân chính dẫn đến trùng node Document:")
    print("1. Lỗi logic sử dụng MERGE không đồng nhất: ")
    print("   - Khi tạo node từ quan hệ (Reference), code dùng: MERGE (p:Document {document_number: rel.target_doc}) và set p.id = 'REF_' + ...")
    print("   - Khi tạo node từ dữ liệu gốc (Full-text), code dùng: MERGE (d:Document {id: row.doc_id})")
    print("   => Do điều kiện MERGE khác nhau, Neo4j không biết gộp chung 2 node này, dẫn đến sinh ra 2 node với cùng document_number.")
    print("2. Lịch sử pháp luật thực tế: Cùng số hiệu (ví dụ '1197/QĐ-UBND') nhưng do các tỉnh/UBND khác nhau hoặc khác năm phát hành. Do bộ pipeline gán ID (UUID) khác nhau nên Neo4j ghi nhận là 2 node riêng biệt.")
    print("3. Rác trong dữ liệu: Khoảng trắng thừa đầu/cuối của document_number chưa được trim().")
    print("4. Case-sensitivity: Sự sai khác giữa hoa/thường trong quá trình gõ OCR/Extract (QĐ vs qđ).")

if __name__ == "__main__":
    main()
