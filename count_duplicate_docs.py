import pandas as pd
from collections import Counter

def count_duplicates(csv_path, output_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    
    # Check if 'document_number' column exists
    if 'document_number' not in df.columns:
        print("Column 'document_number' not found in CSV.")
        return
        
    # Get frequencies of each document number
    # Dropna to ignore empty document numbers if any, but we can also count them
    doc_numbers = df['document_number'].dropna().str.strip()
    
    # Count occurrences of each document number
    counts = doc_numbers.value_counts()
    
    # Now count how many document numbers appear 1 time, 2 times, 3 times...
    frequency_of_counts = counts.value_counts().sort_index()
    
    total_unique_codes = len(counts)
    total_codes = len(doc_numbers)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO THỐNG KÊ SỐ LƯỢNG MÃ VĂN BẢN (DOCUMENT_NUMBER) ===\n")
        f.write(f"Tổng số bản ghi có document_number: {total_codes}\n")
        f.write(f"Tổng số document_number duy nhất (unique): {total_unique_codes}\n")
        f.write("\nPhân bố số lần xuất hiện (trùng lặp):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Số lần xuất hiện (Tần suất)':<30} | {'Số lượng mã (Số lượng document_number)'}\n")
        f.write("-" * 50 + "\n")
        
        for freq, num_docs in frequency_of_counts.items():
            f.write(f"Xuất hiện {freq} lần{' ' * (20 - len(str(freq)))} | {num_docs} mã\n")
            
        f.write("\n\n=== CHI TIẾT CÁC MÃ TRÙNG LẶP NHIỀU NHẤT (Top 20) ===\n")
        top_duplicates = counts[counts > 1].head(20)
        for doc_num, freq in top_duplicates.items():
            f.write(f"Mã '{doc_num}': xuất hiện {freq} lần\n")
            
    print(f"Report saved to {output_path}")

if __name__ == '__main__':
    csv_file = "top_8000_y_te_theo_quyen_luc.csv"
    out_file = "duplicate_counts_report.txt"
    count_duplicates(csv_file, out_file)
