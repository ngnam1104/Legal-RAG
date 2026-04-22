import os
import sys

# Thêm đường dẫn gốc để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.retrieval.graph_db import get_neo4j_driver, run_cypher

def export_graph_report(output_file="results/graph_dump_report.txt"):
    print(f"-> Exporting Graph data to file: {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO TRẠNG THÁI GRAPH DATABASE (NEO4J) ===\n\n")
        
        # 1. Thống kê Node
        f.write("--- 1. Thống kê số lượng Node theo Label ---\n")
        nodes_stat = run_cypher("MATCH (n) RETURN labels(n)[0] as Label, count(*) as Count ORDER BY Count DESC")
        for r in nodes_stat:
            f.write(f"Node [{r['Label']}]: {r['Count']}\n")
        f.write("\n")
        
        # 2. Thống kê Relationship
        f.write("--- 2. Thống kê số lượng Relationship theo Type ---\n")
        rels_stat = run_cypher("MATCH ()-[r]->() RETURN type(r) as Type, count(*) as Count ORDER BY Count DESC")
        for r in rels_stat:
            f.write(f"Rel [{r['Type']}]: {r['Count']}\n")
        f.write("\n")
        
        # 3. Danh sách Document đã nạp (Full Text)
        f.write("--- 3. Danh sách Document đã nạp Full Text ---\n")
        docs = run_cypher("MATCH (d:Document {is_full_text: true}) RETURN d.document_number as num, d.title as title")
        for r in docs:
            f.write(f"- {r['num']}: {r['title']}\n")
        f.write("\n")
        
        # 4. Các mối quan hệ thay thế/sửa đổi (Conflict Analysis)
        f.write("--- 4. Danh sách các quan hệ AMENDS / REPLACES / REPEALS ---\n")
        conflicts = run_cypher("""
            MATCH (s:Document)-[r:AMENDS|REPLACES|REPEALS]->(t:Document)
            RETURN s.document_number as source, type(r) as type, t.document_number as target, r.context as context
        """)
        if not conflicts:
            f.write("[Trống]\n")
        for r in conflicts:
            f.write(f"[{r['source']}] -- {r['type']} --> [{r['target']}]\n")
            f.write(f"   Context: {r['context']}\n")
        f.write("\n")

        # 5. Kiểm tra chi tiết Phantom Hierarchy (MỨC ĐỘ ĐẦY ĐỦ CỦA DỮ LIỆU THAM CHIẾU)
        f.write("--- 5. Kiểm tra độ đầy đủ của Phantom Hierarchy (Văn bản tham chiếu) ---\n")
        # Thống kê tổng quan
        phantom_stats = run_cypher("""
            MATCH (d:Document {is_full_text: false})
            OPTIONAL MATCH (d)<-[:BELONGS_TO]-(art:Article)
            OPTIONAL MATCH (art)<-[:PART_OF]-(cl:Clause)
            RETURN count(DISTINCT d) as total_refs, 
                   count(DISTINCT art) as total_arts, 
                   count(DISTINCT cl) as total_clauses,
                   sum(CASE WHEN art.text IS NOT NULL AND art.text <> '' THEN 1 ELSE 0 END) as arts_with_text
        """)[0]
        f.write(f"Tổng số văn bản tham chiếu (REF_): {phantom_stats['total_refs']}\n")
        f.write(f"Số lượng Điều (Article) được tạo: {phantom_stats['total_arts']} (Trong đó {phantom_stats['arts_with_text']} cái có nội dung text)\n")
        f.write(f"Số lượng Khoản (Clause) được tạo: {phantom_stats['total_clauses']}\n\n")

        # Top 10 văn bản tham chiếu "giàu" thông tin nhất
        f.write("--- 6. Top 10 văn bản tham chiếu có cấu trúc chi tiết nhất ---\n")
        rich_phantoms = run_cypher("""
            MATCH (d:Document {is_full_text: false})<-[:BELONGS_TO]-(art:Article)
            RETURN d.document_number as doc, count(art) as art_count, collect(art.name)[..5] as sample_arts
            ORDER BY art_count DESC LIMIT 10
        """)
        for r in rich_phantoms:
            f.write(f"- {r['doc']}: {r['art_count']} Điều. Ví dụ: {r['sample_arts']}\n")
        f.write("\n")

        # 7. Kiểm tra nốt cô đơn (Isolated Documents)
        f.write("--- 7. Các văn bản tham chiếu 'Rỗng' (Không có cấu trúc thành phần) ---\n")
        f.write("(Đây là các docs chỉ được nhắc tới document_number nhưng LLM không trích xuất được Điều/Khoản cụ thể)\n")
        empty_docs = run_cypher("""
            MATCH (d:Document {is_full_text: false})
            WHERE NOT (d)<-[:BELONGS_TO]-(:Article)
            RETURN d.document_number as num LIMIT 15
        """)
        for r in empty_docs:
            f.write(f"- {r['num']}\n")
        f.write("\n")

        # 8. Phân tích chuỗi quan hệ bắc cầu (Transitive Chain)
        f.write("--- 8. Phân tích chuỗi bắc cầu (A sửa đổi B, B sửa đổi C) ---\n")
        chains = run_cypher("""
            MATCH p = (d1:Document)-[:AMENDS|REPLACES*2..3]->(dn:Document)
            RETURN [n in nodes(p) | n.document_number] as chain, length(p) as depth
            LIMIT 5
        """)
        if not chains:
            f.write("[Không tìm thấy chuỗi bắc cầu nào]\n")
        for r in chains:
            f.write(f"Chuỗi (độ sâu {r['depth']}): {' -> '.join(r['chain'])}\n")
        f.write("\n")

        # 9. Kiểm tra tính nhất quán của Thuộc tính trên Edge
        f.write("--- 9. Mẫu dữ liệu thuộc tính trên Quan hệ (Edge Properties) ---\n")
        edge_data = run_cypher("""
            MATCH (s:Document)-[r:BASED_ON|AMENDS|REPLACES|REPEALS]->(t:Document)
            WHERE r.target_article IS NOT NULL AND r.target_article <> ''
            RETURN s.document_number as src, type(r) as type, t.document_number as tgt, 
                   r.target_article as art, r.target_text as text
            LIMIT 5
        """)
        for r in edge_data:
            f.write(f"Quan hệ: {r['src']} --{r['type']}--> {r['tgt']}\n")
            f.write(f"   Target Location: {r['art']}\n")
            f.write(f"   Sample text in edge: {r['text'][:150]}...\n")
        f.write("\n")
        
        f.write("=== KẾT THÚC BÁO CÁO ===\n")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Success! Report saved at: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    export_graph_report()
