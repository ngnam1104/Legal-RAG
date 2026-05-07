import os
import psycopg2
from dotenv import load_dotenv
# Load cấu hình từ file .env
load_dotenv()
def clear_all_history():
    pg_url = os.environ.get("POSTGRES_URL")
    if not pg_url:
        print("❌ Không tìm thấy POSTGRES_URL trong file .env")
        return
    try:
        conn = psycopg2.connect(pg_url)
        cursor = conn.cursor()
        
        # Xóa dữ liệu trong các bảng (theo thứ tự để tránh lỗi khóa ngoại)
        print("⏳ Đang xóa toàn bộ lịch sử chat...")
        cursor.execute("TRUNCATE TABLE messages RESTART IDENTITY CASCADE;")
        cursor.execute("TRUNCATE TABLE conversation_state CASCADE;")
        cursor.execute("TRUNCATE TABLE sessions CASCADE;")
        
        conn.commit()
        print("✅ Đã xóa sạch toàn bộ phiên chat và tin nhắn trong Database.")
        
        # Nếu dùng Redis, bạn có thể xóa thêm cache ở đây
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            import redis
            r = redis.Redis.from_url(redis_url)
            r.flushdb()
            print("✅ Đã dọn dẹp bộ nhớ đệm Redis.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
if __name__ == "__main__":
    clear_all_history()