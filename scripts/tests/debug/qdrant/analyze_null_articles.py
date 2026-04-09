import os
import sys
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

# Allow importing from root directory (scripts/tests/debug/file.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from backend.config import settings
    QDRANT_URL = settings.QDRANT_URL
    QDRANT_API_KEY = settings.QDRANT_API_KEY
    COLLECTION = settings.QDRANT_COLLECTION
except ImportError:
    QDRANT_URL = "http://localhost:6335"
    QDRANT_API_KEY = None
    COLLECTION = "legal_rag_docs_5000"

def analyze_null_articles():
    print(f"Connecting to Qdrant at: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # 1. Count points with null/missing article_ref
    filter_null = Filter(
        must=[
            IsEmptyCondition(is_empty=PayloadField(key="article_ref"))
        ]
    )
    
    res = client.count(collection_name=COLLECTION, count_filter=filter_null, exact=True)
    null_count = res.count
    print(f"Total points with null/missing 'article_ref': {null_count:,}")
    
    if null_count == 0:
        print("No null articles found.")
        return

    # 2. Sample points to see what's inside
    print("\nSampling first 20 null-article chunks to verify hypothesis...")
    print("-" * 100)
    
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=filter_null,
        limit=20,
        with_payload=True,
        with_vectors=False
    )
    
    for i, p in enumerate(points, 1):
        payload = p.payload
        title = payload.get("title", "N/A")
        doc_num = payload.get("document_number", "N/A")
        text = payload.get("chunk_text", "")
        # Get actual content part (after [NỘI DUNG ...])
        content = text.split("]\n", 2)[-1] if "]\n" in text else text
        content_preview = content[:250].replace("\n", " ").strip()
        
        print(f"#{i} | Doc: {doc_num} | Title: {title[:50]}...")
        print(f"   | Preview: {content_preview}...")
        print("-" * 100)

    # 3. Categorize null-article chunks
    SAMPLE_SIZE = 500
    print(f"\nCategorizing top {SAMPLE_SIZE} points to understand the 'waste'...")
    big_sample, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=filter_null,
        limit=SAMPLE_SIZE,
        with_payload=True
    )
    
    categories = {
        "Table Content (contains |)": 0,
        "Roman Heading (I, II, III)": 0,
        "Numerical Heading (1., 2.)": 0,
        "Chapter/Section (Muc/Chuong)": 0,
        "Administrative / Preamble": 0,
        "Other / Unstructured Content": 0
    }
    
    doc_counts = {}
    
    for p in big_sample:
        payload = p.payload
        t = payload.get("title", "Unknown")
        doc_counts[t] = doc_counts.get(t, 0) + 1
        
        text = payload.get("chunk_text", "").split("]\n", 2)[-1]
        lines = text.splitlines()
        first_lines = "\n".join(lines[:5]).strip()
        all_text = text.lower()
        
        if "|" in text:
            categories["Table Content (contains |)"] += 1
        elif re.search(r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.", first_lines, re.M):
            categories["Roman Heading (I, II, III)"] += 1
        elif re.search(r"^\s*\d+\.", first_lines, re.M):
            categories["Numerical Heading (1., 2.)"] += 1
        elif re.search(r"(?i)^\s*(Muc|Chuong)", first_lines, re.M):
            categories["Chapter/Section (Muc/Chuong)"] += 1
        elif len(all_text) < 500 and any(kw in all_text for kw in ["can cu", "xet de nghi", "ban hanh:", "noi nhan:"]):
            categories["Administrative / Preamble"] += 1
        else:
            categories["Other / Unstructured Content"] += 1
            
    print("-" * 50)
    for cat, count in categories.items():
        pct = (count / len(big_sample) * 100) if big_sample else 0
        print(f"   {cat:<35}: {count:>3} ({pct:>5.1f}%)")
    print("-" * 50)

    print("\nMapping of most affected documents (from sample):")
    sorted_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
    for doc, count in sorted_docs[:10]:
        print(f"   - {count:>3} chunks: {doc}")

if __name__ == "__main__":
    analyze_null_articles()
