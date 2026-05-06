import os
import sys

# Add project root to sys.path
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from dotenv import load_dotenv
load_dotenv(os.path.join(_repo, ".env"))

import json
from backend.retrieval.graph_search import entity_retriever

def main():
    print("=== TEST ENTITY GRAPH RETRIEVER ===")
    
    queries = [
        "Cơ quan nào ban hành Thông tư 19 về vệ sinh?",
        "Xin quy trình cấp phép môi trường tại Bình Dương"
    ]
    
    for q in queries:
        print(f"\n[Query]: {q}")
        
        # 1. Test extraction
        entities = entity_retriever.extract_entities(q)
        print(f"Extracted Entities: {entities}")
        
        if not entities:
            print("No entities found.")
            continue
            
        # 2. Test search
        res = entity_retriever.search_by_entities(entities)
        
        print(f"Found Entities in Graph: {res['found_entities']}")
        print(f"Chunk IDs retrieved: {len(res['chunk_ids'])} chunks")
        print(f"Doc Numbers retrieved: {res['doc_numbers']}")
        print("Graph Context:")
        print(res["graph_context"])
        print("-" * 50)

if __name__ == "__main__":
    main()
