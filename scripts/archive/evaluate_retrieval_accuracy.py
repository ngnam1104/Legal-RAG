import os
import sys
import pandas as pd
import re
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Set path for backend imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval.hybrid_search import retriever
from backend.config import settings
from qdrant_client import models

# ═══════════════════════════════════════════════════════════════
# CẤU HÌNH ĐÁNH GIÁ 
# ═══════════════════════════════════════════════════════════════
RETRIEVAL_LIMIT = 50
USE_DOC_FILTER = True
USE_HYDE_REWRITE = False


def check_doc_exists_in_db(doc_num: str) -> bool:
    """Kiểm tra xem document_number đã có trong Qdrant DB hay chưa"""
    if not doc_num:
        return False
    try:
        count_res = retriever.client.count(
            collection_name=retriever.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_number",
                        match=models.MatchValue(value=doc_num)
                    )
                ]
            )
        )
        return count_res.count > 0
    except Exception as e:
        print(f"Error checking doc '{doc_num}' in DB: {e}")
        return False

def extract_doc_number(text: str) -> str:
    """Extract document number from reference string (e.g. 45/2026/NĐ-CP)"""
    patterns = [
        r"(\d+\/\d+\/[A-Z0-9\-\d\u0110]+)",
        r"(\d+\/[A-Z0-9\-\u0110]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return ""

def extract_article_num(text: str) -> str:
    """Extract article number from reference string (e.g. Điều 2)"""
    m = re.search(r"(?:Điều|Dieu|Điêu)\s+(\d+[A-Za-z0-9\/\-]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""

def extract_clause_num(text: str) -> str:
    """Extract clause number from reference string (e.g. Khoản 1)"""
    m = re.search(r"(?:Khoản|Khoan)\s+(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def is_match(expected_ref: str, hit: Dict[str, Any]) -> dict:
    """
    Check if a retrieved hit matches the expected reference.
    Returns a dict with match result and diagnostic info.
    """
    expected_doc_num = extract_doc_number(expected_ref)
    expected_art_num = extract_article_num(expected_ref)
    
    retrieved_doc_num = hit.get("document_number", "")
    retrieved_art_ref = str(hit.get("article_ref", "") or "")
    retrieved_ref_citation = str(hit.get("reference_citation", "") or "")
    retrieved_chunk_text = str(hit.get("chunk_text", "") or "")
    
    # Combined retrieval text for flexible matching
    all_retrieved_text = f"{retrieved_art_ref} {retrieved_ref_citation} {retrieved_chunk_text}".lower()
    
    # 1. Match Document Number
    doc_match = False
    doc_reason = ""
    if expected_doc_num and retrieved_doc_num:
        if expected_doc_num.lower() in retrieved_doc_num.lower():
            doc_match = True
        else:
            doc_reason = f"doc mismatch: expected={expected_doc_num}, got={retrieved_doc_num}"
    elif not expected_doc_num:
        doc_reason = "no expected doc_num in ground truth"
    else:
        doc_reason = f"expected={expected_doc_num} but hit has no doc_num"

    # 2. Match Article Number
    art_match = False
    art_reason = ""
    if expected_art_num:
        # Check in article_ref, reference_citation, and chunk_text
        art_pattern = f"điều {expected_art_num}".lower()
        if art_pattern in all_retrieved_text:
            art_match = True
        else:
            art_reason = f"art mismatch: expected Điều {expected_art_num}, art_ref={retrieved_art_ref}"
    else:
        # No article number expected, just doc match is enough
        art_match = True
        art_reason = "no article expected"

    matched = doc_match and art_match
    reasons = []
    if not doc_match and doc_reason:
        reasons.append(doc_reason)
    if not art_match and art_reason:
        reasons.append(art_reason)
    
    return {
        "matched": matched,
        "doc_match": doc_match,
        "art_match": art_match,
        "reason": "; ".join(reasons) if reasons else "OK"
    }


def evaluate_single_query(query: str, gt_ref: str, mode: str, filter_doc: Optional[str], rewrite_fn) -> dict:
    """Evaluate a single query and return detailed result."""
    search_query = str(query)
    
    if rewrite_fn:
        try:
            rewrite_data = rewrite_fn(search_query)
            hypothetical = rewrite_data.get("hypothetical_answer")
            if hypothetical and len(hypothetical) > 10:
                search_query = hypothetical
        except Exception:
            pass
    
    t0 = time.perf_counter()
    hits = retriever.search(
        query=search_query, 
        limit=RETRIEVAL_LIMIT,
        use_rerank=False, 
        expand_context=True,
        doc_number=filter_doc,
    )
    latency = time.perf_counter() - t0
    
    # Check all hits
    match_ranks = []
    match_details = []
    for rank, hit in enumerate(hits, 1):
        result = is_match(str(gt_ref), hit)
        if result["matched"]:
            match_ranks.append(rank)
        match_details.append({
            "rank": rank,
            "doc_num": hit.get("document_number", ""),
            "art_ref": str(hit.get("article_ref", "") or "")[:80],
            "is_appendix": hit.get("is_appendix", False),
            "matched": result["matched"],
            "doc_match": result["doc_match"],
            "art_match": result["art_match"],
            "reason": result["reason"]
        })
    
    # Determine verdict
    if match_ranks:
        verdict = "HIT"
        best_rank = min(match_ranks)
    else:
        verdict = "MISS"
        best_rank = -1
    
    # Diagnose failure
    diagnosis = ""
    if verdict == "MISS":
        expected_doc = extract_doc_number(str(gt_ref))
        expected_art = extract_article_num(str(gt_ref))
        
        # Check if any hit has the right doc
        doc_found = any(d["doc_match"] for d in match_details)
        art_found = any(d["art_match"] for d in match_details)
        
        if not doc_found:
            diagnosis = f"DOC_NOT_RETRIEVED (expected {expected_doc})"
        elif doc_found and not art_found:
            diagnosis = f"DOC_OK_ART_MISS (expected Điều {expected_art})"
        else:
            diagnosis = "PARTIAL_MATCH_BUT_CRITERIA_NOT_MET"
    
    return {
        "mode": mode,
        "query": str(query)[:120],
        "expected_ref": str(gt_ref),
        "doc_filter": filter_doc,
        "verdict": verdict,
        "best_rank": best_rank,
        "match_ranks": match_ranks,
        "latency": latency,
        "num_hits": len(hits),
        "diagnosis": diagnosis,
        "top5": match_details[:5],
    }


def print_detailed_result(r: dict, idx: int):
    """Print one query's result in detail."""
    icon = "✅" if r["verdict"] == "HIT" else "❌"
    mode_short = r["mode"].replace("Mode 1: Hỏi đáp trực tiếp", "QA") \
                          .replace("Mode 2: Dẫn dắt tìm kiếm tài liệu", "SEARCH") \
                          .replace("Mode 3: Phát hiện xung đột", "CONFLICT")
    
    print(f"\n{icon} [{idx}] [{mode_short}] {r['query'][:80]}")
    print(f"   Expected : {r['expected_ref']}")
    print(f"   Filter   : doc_number={r['doc_filter'] or 'None'}")
    print(f"   Verdict  : {r['verdict']}  | Best Rank: {r['best_rank'] if r['best_rank'] > 0 else 'N/A'} | Hits: {r['num_hits']} | Latency: {r['latency']:.2f}s")
    
    if r["diagnosis"]:
        print(f"   Diagnosis: ⚠️  {r['diagnosis']}")
    
    # Top-5 hits detail
    print(f"   Top-5 Retrieved:")
    for d in r["top5"]:
        m_icon = "✓" if d["matched"] else "✗"
        app_tag = " [PHỤ LỤC]" if d["is_appendix"] else ""
        print(f"     #{d['rank']} {m_icon} [{d['doc_num']}] {d['art_ref']}{app_tag} | {d['reason']}")


def evaluate_retrieval(csv_path: str, top_k_values=[1, 3, 5, 10]):
    print(f"Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Columns: Mode, Question, Answer, Legal Basis
    modes = df.iloc[:, 0].tolist()
    questions = df.iloc[:, 1].tolist()
    gt_references = df.iloc[:, 3].tolist()
    
    # Lazy-load HyDE Rewriter
    rewrite_fn = None
    if USE_HYDE_REWRITE:
        from backend.agent.utils_legal_qa import rewrite_legal_query
        rewrite_fn = rewrite_legal_query
        print("[CONFIG] HyDE Rewrite: ON")
    else:
        print("[CONFIG] HyDE Rewrite: OFF")
        
    print(f"[CONFIG] Doc Filter   : {'ON' if USE_DOC_FILTER else 'OFF'}")
    print(f"[CONFIG] Retrieval Limit: {RETRIEVAL_LIMIT}")
    
    all_results = []
    skipped = 0
    missing_docs = set()
    
    print(f"\nStarting evaluation on {len(questions)} queries...")
    
    for i, (mode, q, gt_ref) in enumerate(tqdm(
        zip(modes, questions, gt_references), total=len(questions)
    )):
        if pd.isna(q) or not str(q).strip():
            continue
            
        doc_num = extract_doc_number(str(gt_ref))
        if doc_num:
            if not check_doc_exists_in_db(doc_num):
                missing_docs.add(doc_num)
                skipped += 1
                continue
        
        filter_doc = doc_num if (USE_DOC_FILTER and doc_num) else None
        
        result = evaluate_single_query(
            query=q, gt_ref=gt_ref, mode=mode,
            filter_doc=filter_doc, rewrite_fn=rewrite_fn
        )
        all_results.append(result)
    
    # ══════════════════════════════════════════════════════════
    # PRINT DETAILED RESULTS (ALL)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("           CHI TIẾT KẾT QUẢ TỪNG QUERY")
    print("=" * 70)
    
    for idx, r in enumerate(all_results, 1):
        print_detailed_result(r, idx)
    
    # ══════════════════════════════════════════════════════════
    # AGGREGATE METRICS (OVERALL + PER MODE)
    # ══════════════════════════════════════════════════════════
    def compute_metrics(results_subset, label):
        total = len(results_subset)
        if total == 0:
            print(f"\n  [{label}] No data.")
            return {}
        
        metrics = {}
        for k in top_k_values:
            hit_count = sum(1 for r in results_subset if any(rank <= k for rank in r["match_ranks"]))
            metrics[f"Recall@{k}"] = hit_count / total
        
        mrr = sum(1.0 / min(r["match_ranks"]) if r["match_ranks"] else 0 for r in results_subset) / total
        metrics["MRR"] = mrr
        metrics["Avg Latency"] = sum(r["latency"] for r in results_subset) / total
        matched = sum(1 for r in results_subset if r["verdict"] == "HIT")
        unmatched = total - matched
        
        print(f"\n  [{label}] Total: {total} | Hit: {matched} ({matched/total*100:.1f}%) | Miss: {unmatched}")
        for mk, mv in metrics.items():
            if "Latency" in mk:
                print(f"    {mk}: {mv:.4f}s")
            else:
                print(f"    {mk}: {mv:.4f} ({mv*100:.1f}%)")
        
        # Failure breakdown
        misses = [r for r in results_subset if r["verdict"] == "MISS"]
        if misses:
            doc_missing_count = sum(1 for m in misses if "DOC_NOT_RETRIEVED" in m["diagnosis"])
            art_missing_count = sum(1 for m in misses if "DOC_OK_ART_MISS" in m["diagnosis"])
            other_count = len(misses) - doc_missing_count - art_missing_count
            print(f"    Failure Breakdown:")
            print(f"      DOC_NOT_RETRIEVED : {doc_missing_count}")
            print(f"      DOC_OK_ART_MISS   : {art_missing_count}")
            print(f"      OTHER             : {other_count}")
        
        return metrics
    
    print("\n" + "=" * 70)
    print("           TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 70)
    
    if skipped:
        print(f"\n  ⚠️ Skipped {skipped} queries (doc not in DB)")
        if missing_docs:
            print(f"     Missing docs: {', '.join(list(missing_docs)[:10])}")
    
    compute_metrics(all_results, "OVERALL")
    
    # Per-mode breakdown
    mode_names = {
        "Mode 1: Hỏi đáp trực tiếp": "MODE 1: QA",
        "Mode 2: Dẫn dắt tìm kiếm tài liệu": "MODE 2: SEARCH",
        "Mode 3: Phát hiện xung đột": "MODE 3: CONFLICT",
    }
    for mode_key, mode_label in mode_names.items():
        subset = [r for r in all_results if r["mode"] == mode_key]
        compute_metrics(subset, mode_label)
    
    print("\n" + "=" * 70)

    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), "retrieval_eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "hyde_rewrite": USE_HYDE_REWRITE,
                "doc_filter": USE_DOC_FILTER,
                "retrieval_limit": RETRIEVAL_LIMIT,
            },
            "skipped_count": skipped,
            "missing_docs": list(missing_docs),
            "detailed_results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Also save a human-readable TXT report
    txt_output = os.path.join(os.path.dirname(__file__), "retrieval_eval_report.txt")
    with open(txt_output, "w", encoding="utf-8") as f:
        for idx, r in enumerate(all_results, 1):
            icon = "✅" if r["verdict"] == "HIT" else "❌"
            ms = r["mode"].replace("Mode 1: Hỏi đáp trực tiếp", "QA") \
                          .replace("Mode 2: Dẫn dắt tìm kiếm tài liệu", "SEARCH") \
                          .replace("Mode 3: Phát hiện xung đột", "CONFLICT")
            f.write(f"{icon} [{idx}] [{ms}] {r['query'][:80]}\n")
            f.write(f"   Expected : {r['expected_ref']}\n")
            f.write(f"   Verdict  : {r['verdict']} | Rank: {r['best_rank']} | Diag: {r['diagnosis']}\n")
            for d in r["top5"]:
                m = "✓" if d["matched"] else "✗"
                f.write(f"     #{d['rank']} {m} [{d['doc_num']}] {d['art_ref']} | {d['reason']}\n")
            f.write("\n")
    print(f"Human-readable report: {txt_output}")


if __name__ == "__main__":
    csv_file = r"d:\iCOMM\Legal-RAG\Bộ test Hỏi đáp pháp luật về dữ liệu cá nhân.csv"
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
    else:
        evaluate_retrieval(csv_file)
