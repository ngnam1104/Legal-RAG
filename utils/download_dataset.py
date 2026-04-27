import os
import pandas as pd
from datasets import load_dataset
from rich.console import Console

def main():
    console = Console()
    dataset_name = "nhn309261/vietnamese-legal-documents"
    subset = "metadata"
    cache_path = "D:/huggingface_cache" 
    
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tải dữ liệu
    console.print(f"[bold green]Đang tải dataset {dataset_name} ({subset})...[/bold green]")
    try:
        try:
            ds = load_dataset(dataset_name, name=subset, split="data", cache_dir=cache_path)
        except ValueError:
            ds = load_dataset(dataset_name, name=subset, split="train", cache_dir=cache_path)
    except Exception as e:
        console.print(f"[bold red]Lỗi khi tải dataset: {e}[/bold red]")
        return

    # 2. Lọc lĩnh vực "Thể thao - Y tế"
    console.print("[bold yellow]Đang lọc các văn bản thuộc lĩnh vực 'Thể thao - Y tế'...[/bold yellow]")
    ds_filtered = ds.filter(lambda row: row["legal_sectors"] is not None and "Thể thao - Y tế" in str(row["legal_sectors"]))
    
    # 3. Chuyển sang Pandas DataFrame để dễ dàng sắp xếp (Sorting)
    console.print("[bold yellow]Đang chuyển đổi sang Pandas để phân cấp quyền lực...[/bold yellow]")
    df = ds_filtered.to_pandas()

    # Định nghĩa từ điển trọng số quyền lực (Số càng nhỏ, quyền lực càng cao)
    hierarchy_weights = {
        "Hiến pháp": 1,
        "Bộ luật": 2,
        "Luật": 3,
        "Pháp lệnh": 4,
        "Lệnh": 6,
        "Nghị định": 7,
        "Quyết định": 8,
        "Thông tư liên tịch": 9,
        "Thông tư": 10,
        "Chỉ thị": 11
    }

    # Hàm map trọng số, nếu loại văn bản không có trong danh sách thì cho trọng số 99 (thấp nhất)
    df['hierarchy_rank'] = df['legal_type'].map(lambda x: hierarchy_weights.get(str(x).strip(), 99))

    # Chuyển cột ngày tháng sang định dạng datetime để sort (Văn bản mới nhất lên trên)
    # coerce để biến các ngày bị lỗi format thành NaT (Not a Time)
    df['issuance_date'] = pd.to_datetime(df['issuance_date'], errors='coerce')

    # 4. Sắp xếp: Ưu tiên 1 là hierarchy_rank (nhỏ lên trước), Ưu tiên 2 là issuance_date (mới nhất lên trước)
    console.print("[bold yellow]Đang sắp xếp văn bản từ quyền lực cao xuống thấp, và lấy 8000 văn bản đầu tiên...[/bold yellow]")
    df_sorted = df.sort_values(by=['hierarchy_rank', 'issuance_date'], ascending=[True, False])

    # Cắt lấy 8000 dòng đầu tiên (hoặc lấy tất cả nếu ít hơn 8000)
    df_top_8000 = df_sorted.head(8000)
    
    # Xóa cột trọng số tạm thời nếu không cần thiết
    df_top_8000 = df_top_8000.drop(columns=['hierarchy_rank'])

    # 5. Xuất ra file CSV
    csv_path = os.path.join(output_dir, "top_8000_y_te_theo_quyen_luc.csv")
    df_top_8000.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    console.print(f"[bold green]Hoàn tất! Đã lưu {len(df_top_8000)} văn bản ra file: {csv_path}[/bold green]")

if __name__ == "__main__":
    main()