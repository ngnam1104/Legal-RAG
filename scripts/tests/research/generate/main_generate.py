import os
import sys

# Cấu hình đường dẫn: 4 cấp lên tới Root (scripts/tests/research/generate/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESEARCH_DIR = os.path.join(ROOT_DIR, "scripts", "tests", "research")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from config import MODE1_OUTPUT_PATH, MODE2_OUTPUT_PATH, MODE3_OUTPUT_PATH
from generate.data_loader import load_clusters_from_qdrant, load_clusters_from_document
from generate.generator import generate_mode_1_search, generate_mode_2_qa, generate_mode_3_conflict

def save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"📁 Đã lưu {len(data)} records vào: {path}")

def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu kiểm thử giả định (Synthetic Dataset) cho hệ thống Legal RAG.")
    parser.add_argument("--source", type=str, choices=["qdrant", "file"], default="qdrant", help="Nguồn dữ liệu đầu vào: 'qdrant' (mặc định) hoặc 'file'.")
    parser.add_argument("--path", type=str, help="Đường dẫn đến file tài liệu (chỉ dùng khi --source file).")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng cụm văn bản lấy mẫu từ Qdrant (mặc định 10). Mỗi cụm có thể sinh nhiều mẫu.")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 Legal RAG - Synthetic Dataset Generator Pipeline")
    print("="*60)
    
    # BƯỚC 1: Tải Data (Qdrant hoặc Upload)
    clusters = []
    if args.source == "qdrant":
        clusters = load_clusters_from_qdrant(limit=args.limit, cluster_size=3)
    else:
        if not args.path:
            print("❌ Lỗi: Vui lòng cung cấp đường dẫn tài liệu qua cờ --path khi dùng source 'file'")
            sys.exit(1)
        clusters = load_clusters_from_document(args.path, cluster_size=3)
        
    if not clusters:
        print("❌ Không có dữ liệu để xử lý. Pipeline dừng.")
        sys.exit(1)
        
    # Chuẩn bị nơi lưu trữ kết quả cuối cùng
    all_mode1 = []
    all_mode2 = []
    all_mode3 = []

    print(f"\n🚀 TIẾN HÀNH SINH DỮ LIỆU TỪ {len(clusters)} CỤM TÀI LIỆU...")
    print("⚠️ Có thể mất thời gian tuỳ thuộc vào tốc độ phản hồi API...")
    
    import time

    # BƯỚC 2 & 3: Lặp qua từng cluster và gọi Generator
    for i in tqdm(range(len(clusters)), desc="Processing Clusters"):
        cluster = clusters[i]
        
        # Generator 1: Search Queries (Hybrid Search)
        m1_data = generate_mode_1_search(cluster)
        for item in m1_data:
            if isinstance(item, dict):
                if "expected_chunk_ids" not in item:
                    item["expected_chunk_ids"] = [c.get("chunk_id", "") for c in cluster]
                all_mode1.append(item)
                
        time.sleep(3) # Tránh Rate Limit 
                
        # Generator 2: QA & Scenarios
        m2_data = generate_mode_2_qa(cluster)
        for item in m2_data:
            if isinstance(item, dict):
                if "expected_chunk_ids" not in item:
                    item["expected_chunk_ids"] = [c.get("chunk_id", "") for c in cluster]
                all_mode2.append(item)
                
        time.sleep(3) # Tránh Rate Limit 
                
        # Generator 3: Conflict Detection
        m3_data = generate_mode_3_conflict(cluster)
        for item in m3_data:
            if isinstance(item, dict):
                if "expected_chunk_ids" not in item:
                    item["expected_chunk_ids"] = [c.get("chunk_id", "") for c in cluster]
                all_mode3.append(item)
                
        time.sleep(15) # Nghỉ giữa các cụm để tránh bão hoà Groq Free Tier TPM
                
    # BƯỚC CUỐI: Lưu Data
    print("\n✅ HOÀN TẤT SINH DỮ LIỆU. KẾT QUẢ:")
    print(f"  - Mode 1 (Search Queries) : {len(all_mode1)} items")
    print(f"  - Mode 2 (QA & Scenarios) : {len(all_mode2)} items")
    print(f"  - Mode 3 (Conflict Check) : {len(all_mode3)} items")
    
    save_json(all_mode1, MODE1_OUTPUT_PATH)
    save_json(all_mode2, MODE2_OUTPUT_PATH)
    save_json(all_mode3, MODE3_OUTPUT_PATH)
    print("\n🎉 Toàn bộ quy trình sinh test case dữ liệu đã kết thúc.")

if __name__ == "__main__":
    main()
