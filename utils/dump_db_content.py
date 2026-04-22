import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from qdrant_client import models
from backend.retrieval.vector_db import client as qdrant_client
from backend.config import settings
from backend.retrieval.graph_db import get_neo4j_driver

output_file = "results/dump_output.txt"
# Đảm bảo thư mục kết quả tồn tại
os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
collection_name = settings.QDRANT_COLLECTION

def get_sample_doc_numbers(limit=10):
    driver = get_neo4j_driver()
    if not driver:
        return []
    try:
        with driver.session() as session:
            query = "MATCH (d:Document) RETURN d.document_number AS doc_num LIMIT $limit"
            res = session.run(query, limit=limit).data()
            return [r['doc_num'] for r in res if r['doc_num']]
    except Exception as e:
        print(f"Error fetching sample docs: {e}")
        return []
    finally:
        driver.close()

doc_numbers = get_sample_doc_numbers(10)

if not doc_numbers:
    print("No documents found in Neo4j to dump.")
    sys.exit(1)

def dump_qdrant_v2(f, d_num):
    f.write("="*60 + "\n")
    f.write(f"1. QDRANT (VECTOR DB) DUMP FOR {d_num}\n")
    f.write("="*60 + "\n")
    try:
        # Match Text to allow partial/flex match
        must_cond = [models.FieldCondition(key="document_number", match=models.MatchText(text=d_num))]
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=must_cond),
            with_payload=True,
            limit=15 # Giới hạn 15 chunk mỗi văn bản
        )
        if points:
            f.write(f"Total chunks found in Qdrant: {len(points)}\n\n")
            for i, p in enumerate(points):
                f.write(f"--- Chunk {i+1} (ID: {p.id}) ---\n")
                payload = p.payload or {}
                f.write(f"Title: {payload.get('title')}\n")
                f.write(f"Article Ref: {payload.get('article_ref')}\n")
                f.write(f"Is Appendix: {payload.get('is_appendix')}\n")
                f.write(f"Chunk Index: {payload.get('chunk_index')}\n")
                text = payload.get('chunk_text') or payload.get('text', '')
                f.write(f"Content Preview (500 chars):\n{text[:500]}...\n\n")
        else:
            f.write("NO CHUNKS FOUND IN QDRANT!\n\n")
    except Exception as e:
        f.write(f"Error querying Qdrant: {e}\n\n")

def dump_neo4j_v2(f, d_num):
    f.write("="*60 + "\n")
    f.write(f"2. NEO4J (GRAPH DB) DUMP FOR {d_num}\n")
    f.write("="*60 + "\n")
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            # Document Node
            f.write("--- DOCUMENT NODE ---\n")
            doc_query = "MATCH (d:Document) WHERE d.document_number = $d_num RETURN d"
            doc_res = session.run(doc_query, d_num=d_num).data()
            if doc_res:
                for record in doc_res:
                    node = record['d']
                    f.write(json.dumps(dict(node), ensure_ascii=False, indent=2) + "\n")
            else:
                f.write("Document node not found.\n")
            f.write("\n")

            # Article & Clause Nodes
            f.write("--- HIERARCHY (Top 20) ---\n")
            hier_query = """
            MATCH (d:Document)<-[:BELONGS_TO]-(a:Article)
            WHERE d.document_number = $d_num
            OPTIONAL MATCH (a)<-[:PART_OF]-(c:Clause)
            RETURN a.name AS Article, a.text AS ArticleText, c.name AS Clause, c.text AS ClauseText
            ORDER BY a.name, c.name
            LIMIT 20
            """
            hier_res = session.run(hier_query, d_num=d_num).data()
            if hier_res:
                for r in hier_res:
                    art_clean = r.get('ArticleText', '').replace('\n', ' ')[:80] if r.get('ArticleText') else 'None'
                    clause_clean = r.get('ClauseText', '').replace('\n', ' ')[:80] if r.get('ClauseText') else 'None'
                    f.write(f"- Article: {r['Article']} | Clause: {r['Clause']} \n  ArtText: {art_clean}...\n  ClauseText: {clause_clean}...\n")
            else:
                f.write("No Articles/Clauses found.\n")
            
    except Exception as e:
        f.write(f"Error querying Neo4j: {e}\n\n")
    finally:
        driver.close()

with open(output_file, "w", encoding="utf-8") as f:
    for doc_number in doc_numbers:
        f.write("\n" + "#"*80 + "\n")
        f.write(f"### DUMPING DOCUMENT: {doc_number}\n")
        f.write("#"*80 + "\n")
        dump_qdrant_v2(f, doc_number)
        dump_neo4j_v2(f, doc_number)

print(f"Successfully dumped 10 documents to {output_file}")
