import os
import sys
from typing import List, Dict, Any

# Cấu hình đường dẫn: 5 cấp lên tới Root (scripts/tests/data/research_gen/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from config import QDRANT_URL, QDRANT_COLLECTION

def load_clusters_from_qdrant(limit: int = 10, cluster_size: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Kết nối đến Qdrant local, lấy ngẫu nhiên {limit} chunks gốc (seed).
    Với mỗi chunk gốc, dùng vector của nó để tìm thêm {cluster_size - 1} chunks liên quan
    nhất có cùng chủ đề (có thể từ các luật khác nhau) để tạo thành một nhóm tài liệu (cluster).
    """
    from qdrant_client import QdrantClient
    print(f"🔄 Đang kết nối Qdrant tại {QDRANT_URL} để tạo {limit} clusters...")
    
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        # Scroll API to get N seed records with vectors
        records, next_page = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        clusters = []
        for record in records:
            if not record.payload or not record.vector:
                continue
                
            # Tạo cluster với chunk gốc đầu tiên
            cluster = [record.payload]
            
            # Nếu cần tìm thêm chunks cùng chủ đề
            if cluster_size > 1:
                # Tìm kiếm các chunks gần nhất dựa trên vector
                query_vector = record.vector.get("dense") if isinstance(record.vector, dict) else record.vector
                using_name = "dense" if isinstance(record.vector, dict) and "dense" in record.vector else None
                
                search_result = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=query_vector,
                    using=using_name,
                    limit=cluster_size, # Lấy cả bản thân nó và thêm 2 cái nữa
                    with_payload=True,
                )
                
                # Biến list top-k trả về thành danh sách payload tránh trùng lặp
                seen_ids = {record.id}
                for hit in search_result.points:
                    if hit.id not in seen_ids and hit.payload:
                        cluster.append(hit.payload)
                        seen_ids.add(hit.id)
                        
            clusters.append(cluster)
                
        print(f"✅ Đã tạo thành công {len(clusters)} clusters từ Qdrant (mỗi cluster có {cluster_size} tài liệu cùng chủ đề).")
        return clusters
        
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return []

def load_clusters_from_document(file_path: str, cluster_size: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Đọc một file PDF/DOCX/TXT tải lên, dùng parser của hệ thống để phân rã ra thành các chunk.
    Gom các chunk liên tiếp thành từng cụm (cluster) để tạo ngữ cảnh rộng hơn.
    """
    from backend.utils.document_parser import parser
    import uuid
    
    print(f"📄 Đang xử lý file tải lên: {file_path}")
    if not os.path.exists(file_path):
        print("❌ Không tìm thấy file.")
        return []
        
    try:
        raw_chunks = parser.parse_and_chunk(file_path)
        if not raw_chunks:
            print("❌ File rỗng hoặc không thể parse.")
            return []
            
        extracted_data = []
        filename = os.path.basename(file_path)
        
        for idx, text_chunk in enumerate(raw_chunks):
            payload = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": filename,
                "document_number": filename.split('.')[0],
                "title": f"Tài liệu test: {filename}",
                "chunk_text": text_chunk,
                "legal_type": "Quy chế/Nội bộ",
                "reference_tag": f"Đoạn {idx+1}",
                "source": "Local Upload"
            }
            extracted_data.append(payload)
            
        clusters = [extracted_data[i:i + cluster_size] for i in range(0, len(extracted_data), cluster_size)]
            
        print(f"✅ Đã parse thành công {len(clusters)} clusters từ file.")
        return clusters
        
    except Exception as e:
        print(f"❌ Lỗi xử lý file (hãy chắc chắn file đúng định dạng): {e}")
        return []
