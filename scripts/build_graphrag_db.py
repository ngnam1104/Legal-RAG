import os
import re
import uuid
import torch
import pandas as pd
from typing import Any, Dict, List, Tuple
from datasets import load_dataset
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
import argparse

# --- CẤU HÌNH ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "legal_graphrag_bge"
DENSE_MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SECTORS = ["Tài nguyên - Môi trường", "Xây dựng - Đô thị", "Bất động sản"]
VALID_TYPES = ["Luật", "Bộ luật", "Nghị định", "Thông tư"]
TARGET_YEARS = [2014, 2020, 2024]
MAX_YEAR = 2025
CHUNKS_PER_SECTOR = 3500

class LocalBGEHybridEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"), device=device)

    @staticmethod
    def _to_sparse_vector(weights: Dict[str, float]) -> models.SparseVector:
        if not weights:
            return models.SparseVector(indices=[], values=[])
        pairs = [(int(k), float(v)) for k, v in weights.items() if float(v) != 0.0]
        pairs.sort(key=lambda x: x[0])
        return models.SparseVector(indices=[i for i, _ in pairs], values=[v for _, v in pairs])

    def encode_hybrid(self, texts: List[str], batch_size: int = 16):
        if isinstance(texts, str): texts = [texts]
        out = self.model.encode(texts, batch_size=(128 if torch.cuda.is_available() else batch_size), max_length=2048, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = out["dense_vecs"]
        dense_list = dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else [list(v) for v in dense_vecs]
        sparse_list = [self._to_sparse_vector(w) for w in out["lexical_weights"]]
        return dense_list, sparse_list

# --- HELPER FUNCTIONS ---
def compact_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()

def split_sector_list(value: Any) -> List[str]:
    if pd.isna(value) or not value: return []
    raw_text = ", ".join(str(x) for x in value if str(x).strip()) if isinstance(value, list) else str(value).strip()
    if not raw_text or raw_text.lower() == "nan": return []
    return [p.strip() for p in re.split(r",\s*", raw_text) if p.strip()]

def get_year(date_str: Any) -> int:
    if pd.isna(date_str) or not str(date_str).strip(): return 0
    date_str = str(date_str).strip()
    m = re.search(r"\b(19|20)\d{2}\b", date_str)
    return int(m.group(0)) if m else 0

# --- CHUNKER Y HỆT REINDEX_APPENDIX_ARTICLES ---
class AdvancedLegalChunker:
    def __init__(self):
        self.true_appendix_pattern = re.compile(
            r"(?m)^\s*("
            r"(?i:Mẫu\s+số|Mẫu|Biểu\s+mẫu)[\s\d\w\.\-\:]*|"
            r"(?i:Phụ\s+lục|Phu\s+luc|Danh\s+mục|Danh\s+muc|Bảng\s+biểu|Bang\s+bieu)\s*(?:\d+|[IVXLCDM]+)\b.*|"
            r"PHỤ LỤC\b.*|PHU LUC\b.*|"
            r"DANH MỤC\b.*|DANH MUC\b.*|"
            r"BẢNG BIỂU\b.*|BANG BIEU\b.*"
            r")$"
        )
        self.substantive_title_pattern = re.compile(r"(?im)^\s*(QUY ĐỊNH|QUY CHẾ|PHƯƠNG ÁN|ĐIỀU LỆ|CHƯƠNG TRÌNH|HƯỚNG DẪN|NỘI QUY|KẾ HOẠCH|CHIẾN LƯỢC)\b.*$")
        self.chapter_pattern = re.compile(r"(?im)^\s*(Chương\s+[IVXLCDM0-9]+)\s*(.*)$")
        self.article_pattern = re.compile(r"(?im)^\s*(Điều\s+\d+[A-Za-z0-9\/\-]*)\s*[\.\:\-]?\s*(.*)$")
        self.clause_pattern = re.compile(r"(?im)^\s*(Khoản\s+\d+[\.\:\-]?)\s*(.*)$|^\s*(\d+[\.\)])\s*(.*)$")
        self.point_pattern = re.compile(r"(?im)^\s*([a-zđ]\s*[\)\.])\s*(.*)$")
        
        self.legal_basis_line_pattern = re.compile(r"(?im)^\s*căn cứ\b.*$")
        self.legal_ref_pattern = re.compile(r"(?i)\b(Hiến pháp|Bộ luật|Luật|Nghị quyết|Pháp lệnh|Nghị định|Thông tư)\b([^.;\n]*)")

        self.final_article_trigger = re.compile(
            r"(?i)^("
            r"(?:Điều\s+\d+[\.\:\-]?\s*)?(?:điều\s+khoản\s+thi\s+hành|hiệu\s+lực\s+thi\s+hành|tổ\s+chức\s+thực\s+hiện|trách\s+nhiệm\s+thi\s+hành)|"
            r".*chịu\s+trách\s+nhiệm\s+thi\s+hành|"
            r".*có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày"
            r")"
        )
        self.footer_pattern = re.compile(
            r"(?i)^\s*(nơi\s+nhận|kính\s+gửi)[\:\.]?|"
            r"^\s*(TM\.|KT\.|Q\.|TL\.|TUQ\.)?\s*"
            r"("
            r"CHÍNH\s+PHỦ|UBND|ỦY\s+BAN\s+NHÂN\s+DÂN|"
            r"BỘ\s+TRƯỞNG|CHỦ\s+TỊCH|THỨ\s+TRƯỞNG|GIÁM\s+ĐỐC|TỔNG\s+GIÁM\s+ĐỐC|"
            r"CỤC\s+TRƯỞNG|TỔNG\s+CỤC\s+TRƯỞNG|CHÁNH\s+VĂN\s+PHÒNG|"
            r"CHÁNH\s+ÁN|VIỆN\s+TRƯỞNG|TỔNG\s+KIỂM\s+TOÁN|CHỦ\s+NHIỆM|TỔNG\s+BÍ\s+THƯ"
            r")\b"
        )

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "unknown"

    def _extract_doc_number(self, text: str) -> str:
        for p in [r"(?i)(?:số\s*)?(\d+\/\d+(?:\/[A-Z0-9Đ\-]+)?)", r"(?i)(\d+\/[A-Z0-9Đ\-]+)"]:
            m = re.search(p, text or "")
            if m: return compact_whitespace(m.group(1))
        return ""

    def _extract_year(self, text: str) -> str:
        m = re.search(r"\b(19|20)\d{2}\b", text or "")
        return m.group(0) if m else "unknown"

    def _parse_legal_basis_line(self, raw_line: str):
        refs = []
        line = compact_whitespace(raw_line)
        for m in self.legal_ref_pattern.finditer(line):
            raw_type = compact_whitespace(m.group(1))
            tail = compact_whitespace(m.group(2))
            full_ref = compact_whitespace(f"{raw_type} {tail}")
            doc_number = self._extract_doc_number(full_ref)
            if not doc_number: continue
            refs.append({
                "doc_type": raw_type, 
                "doc_number": doc_number,
                "doc_title": full_ref,
            })
        return refs

    def _extract_legal_basis_metadata(self, content: str) -> List[dict]:
        preamble = "\n".join((content or "").splitlines()[:80])
        all_refs = []
        for line in preamble.splitlines():
            if self.legal_basis_line_pattern.match(line):
                all_refs.extend(self._parse_legal_basis_line(line))
        return all_refs

    @staticmethod
    def _parse_signer(signer_raw: Any):
        s = str(signer_raw or "")
        if ":" not in s: return s, None
        parts = s.split(":")
        try: return parts[0].strip(), parts[1].strip()
        except: return parts[0].strip(), None

    def process_document(self, content: str, metadata: Dict[str, Any]) -> tuple:
        content = str(content or "").replace("\r\n", "\n").strip()
        doc_id = str(metadata.get("id"))
        doc_number = metadata.get("document_number", "N/A")
        signer_name, signer_id = self._parse_signer(metadata.get("signers", ""))
        promulgation_date = metadata.get("issuance_date", "")
        basis_refs = self._extract_legal_basis_metadata(content)

        chunks = []
        doc_meta = {
            "document_id": doc_id,
            "document_number": doc_number,
            "title": metadata.get("title", "N/A"),
            "legal_type": metadata.get("legal_type", "N/A"),
            "legal_sectors": metadata.get("legal_sectors_list", []),
            "issuing_authority": metadata.get("issuing_authority", "N/A"),
            "signer_name": signer_name,
            "signer_id": signer_id,
            "promulgation_date": promulgation_date,
            "basis_refs": basis_refs,
            "url": metadata.get("url", "")
        }

        # Simplified state machine for this exact task
        lines = content.splitlines()
        current_appendix_buffer = []
        article_preamble = []
        current_clauses_data = []
        
        is_in_appendix = False
        found_final_article = False
        found_administrative_footer = False
        
        current_chapter = None
        current_article_ref = None
        current_article_title = ""
        current_clause_preamble = ""
        
        global_chunk_idx = 0
        current_article_idx = 0
        current_appendix_idx = 0
        current_appendix_name = ""

        def flush_buffer():
            nonlocal global_chunk_idx, current_clauses_data, article_preamble, current_appendix_buffer, current_clause_preamble
            if not current_clauses_data and not article_preamble and not current_appendix_buffer: return

            parts = []
            if is_in_appendix:
                parts.extend(current_appendix_buffer)
                cl_refs = []
            else:
                if article_preamble: parts.append("\n".join(article_preamble))
                cl_refs = [c["ref"] for c in current_clauses_data]
                if current_clauses_data:
                    if current_clause_preamble and "tiếp theo" in cl_refs[0]:
                        parts.append(current_clause_preamble)
                    parts.extend([c["text"] for c in current_clauses_data])

            text_content = "\n".join(parts).strip()
            if not text_content: return
            
            if is_in_appendix and len(text_content.replace("|", "").strip()) < 10:
                current_appendix_buffer.clear(); return
            if not is_in_appendix and len(text_content) < 30:
                article_preamble.clear(); current_clauses_data.clear(); return

            global_chunk_idx += 1
            if is_in_appendix:
                chunk_id_val = f"{doc_id}::appendix::{current_appendix_idx}::c{global_chunk_idx}"
                cl_ref_meta = None
                current_appendix_buffer.clear()
            else:
                chunk_id_val = f"{doc_id}::article::{current_article_idx}::c{global_chunk_idx}"
                cl_ref_meta = cl_refs[0].replace(" (tiếp theo)", "").strip() if cl_refs else None

            chunks.append({
                "chunk_id": chunk_id_val,
                "text_to_embed": text_content, # Simplified embedded text for exactness
                "metadata": {
                    "document_id": doc_id,
                    "chunk_index": global_chunk_idx,
                    "chapter_ref": current_chapter,
                    "article_ref": f"{current_article_ref}. {current_article_title}".strip(". ") if current_article_ref else None,
                    "clause_ref": cl_ref_meta,
                    "is_appendix": is_in_appendix
                }
            })

        for line in lines:
            line = line.strip()
            if not line: continue

            if not is_in_appendix and not found_final_article:
                if self.final_article_trigger.search(line): found_final_article = True

            if not is_in_appendix:
                if self.footer_pattern.match(line) or (line.isupper() and len(line) > 5 and found_final_article):
                    flush_buffer()
                    found_administrative_footer = True
                    article_preamble = [line]; current_article_ref = None; current_clauses_data = []
                    continue

            m_app = self.true_appendix_pattern.match(line)
            if m_app:
                if found_final_article or found_administrative_footer or is_in_appendix:
                    flush_buffer()
                    is_in_appendix = True
                    current_appendix_idx += 1
                    current_appendix_name = compact_whitespace(line)
                    current_chapter, current_article_ref, current_article_title = None, None, ""
                continue

            if self.substantive_title_pattern.match(line) and found_administrative_footer:
                flush_buffer()
                is_in_appendix = False
                article_preamble = [line]; current_article_ref = None; found_final_article = False
                continue

            if not is_in_appendix and found_final_article and line.startswith("|"):
                flush_buffer()
                current_clauses_data = []; article_preamble = []; is_in_appendix = True
                current_appendix_idx += 1; current_appendix_name = "Bảng biểu"
                current_chapter, current_article_ref, current_article_title = None, None, ""
                current_appendix_buffer.append(line)
                continue

            if is_in_appendix:
                current_appendix_buffer.append(line)
                if sum(len(s) for s in current_appendix_buffer) > 5000:
                    flush_buffer()
                continue

            m_ch = self.chapter_pattern.match(line)
            if m_ch:
                flush_buffer()
                is_in_appendix = False; current_clauses_data = []; article_preamble = []
                current_chapter = compact_whitespace(f"{m_ch.group(1)}. {m_ch.group(2)}")
                continue

            m_ar = self.article_pattern.match(line)
            if m_ar:
                article_remainder = m_ar.group(2).strip()
                if len(article_remainder) > 300 and not article_remainder.isupper(): pass
                else:
                    flush_buffer()
                    is_in_appendix = False; current_clauses_data = []; article_preamble = []
                    current_article_idx += 1; current_article_ref = m_ar.group(1).strip()
                    current_article_title = article_remainder[:300] + "..." if len(article_remainder) > 300 else article_remainder
                    if article_remainder: article_preamble.append(article_remainder)
                    continue

            m_cl = self.clause_pattern.match(line)
            if m_cl and current_article_ref:
                if current_clauses_data: flush_buffer()
                current_clauses_data = []
                cl_ref = compact_whitespace(m_cl.group(1) or m_cl.group(3))
                cl_text = compact_whitespace(m_cl.group(2) or m_cl.group(4))
                full_preamble = f"{cl_ref} {cl_text}".strip() if cl_text else cl_ref
                current_clause_preamble = full_preamble[:300] + "..." if len(full_preamble) > 300 else full_preamble
                current_clauses_data.append({"ref": cl_ref, "text": full_preamble})
                continue

            m_pt = self.point_pattern.match(line)
            if m_pt and current_clauses_data and not is_in_appendix:
                point_ref = compact_whitespace(m_pt.group(1))
                point_text = compact_whitespace(m_pt.group(2))
                current_len = sum(len(c["text"]) for c in current_clauses_data) + sum(len(p) for p in article_preamble)
                if current_len > 800:
                    last_ref = current_clauses_data[-1]["ref"].replace(" (tiếp theo)", "")
                    flush_buffer()
                    current_clauses_data = [{"ref": f"{last_ref} (tiếp theo)", "text": f"{point_ref} {point_text}"}]
                else:
                    current_clauses_data[-1]["text"] += f"\n{point_ref} {point_text}"
                continue

            if len(line) > 5000 and not is_in_appendix:
                flush_buffer()
                is_in_appendix = True
                current_appendix_buffer.append(line)
                continue

            if current_clauses_data: current_clauses_data[-1]["text"] += f"\n{line}"
            elif current_article_ref: article_preamble.append(line)
            else: article_preamble.append(line)

        flush_buffer()
        return doc_meta, chunks


def build_neo4j(driver, doc_meta, chunks):
    with driver.session() as session:
        # Merge Document
        session.run("""
            MERGE (d:Document {id: $doc_id})
            SET d.document_number = $doc_num,
                d.title = $title,
                d.legal_type = $l_type,
                d.promulgation_date = $p_date,
                d.url = $url
            """, 
            doc_id=doc_meta["document_id"], doc_num=doc_meta["document_number"],
            title=doc_meta["title"], l_type=doc_meta["legal_type"],
            p_date=doc_meta["promulgation_date"], url=doc_meta["url"]
        )

        # Merge Authority & Signer
        if doc_meta["issuing_authority"] and doc_meta["issuing_authority"] != "N/A":
            session.run("""
                MERGE (a:Authority {name: $auth_name})
                MERGE (a)-[:ISSUED]->(d:Document {id: $doc_id})
            """, auth_name=doc_meta["issuing_authority"], doc_id=doc_meta["document_id"])
            
        if doc_meta["signer_name"]:
            session.run("""
                MERGE (s:Signer {name: $signer_name})
                MERGE (s)-[:SIGNED]->(d:Document {id: $doc_id})
            """, signer_name=doc_meta["signer_name"], doc_id=doc_meta["document_id"])

        # Sectors
        for sec in doc_meta["legal_sectors"]:
            session.run("""
                MERGE (s:Sector {name: $sec_name})
                MERGE (d:Document {id: $doc_id})-[:HAS_SECTOR]->(s)
            """, sec_name=sec, doc_id=doc_meta["document_id"])
            
        # Parent Documents
        for ref in doc_meta["basis_refs"]:
            parent_id = f"REF_{ref['doc_number']}" # Using doc_number as reference
            session.run("""
                MERGE (p:ParentDocument {id: $ref_id})
                SET p.document_number = $doc_num, p.legal_type = $doc_type
                MERGE (d:Document {id: $doc_id})-[:BASED_ON]->(p)
            """, ref_id=parent_id, doc_num=ref['doc_number'], 
               doc_type=ref['doc_type'], doc_id=doc_meta["document_id"])

        # Chunks
        for ch in chunks:
            session.run("""
                MERGE (c:Chunk {id: $chunk_id})
                SET c.chunk_index = $chunk_idx,
                    c.chapter_ref = $chap_ref,
                    c.article_ref = $art_ref,
                    c.clause_ref = $cl_ref,
                    c.is_appendix = $is_app
                MERGE (c)-[:BELONGS_TO]->(d:Document {id: $doc_id})
            """, chunk_id=ch["chunk_id"], chunk_idx=ch["metadata"]["chunk_index"],
               chap_ref=ch["metadata"]["chapter_ref"], art_ref=ch["metadata"]["article_ref"],
               cl_ref=ch["metadata"]["clause_ref"], is_app=ch["metadata"]["is_appendix"],
               doc_id=doc_meta["document_id"])

def main():
    print("Loading dataset...")
    ds_metadata = load_dataset("nhn309261/vietnamese-legal-documents", "metadata", split="data")
    ds_content = load_dataset("nhn309261/vietnamese-legal-documents", "content", split="data")
    
    df = ds_metadata.to_pandas()
    df["legal_sectors_list"] = df["legal_sectors"].apply(split_sector_list)
    df["promulgation_year"] = df["issuance_date"].apply(get_year)
    
    valid_df = df[df["legal_type"].isin(VALID_TYPES) & (df["promulgation_year"] <= MAX_YEAR)]
    
    content_map = {str(row["id"]): row.get("text", row.get("content", "")) for row in ds_content}
    
    print("Filtering documents based on sectors...")
    selected_docs = []
    
    for sector in TARGET_SECTORS:
        sector_df = valid_df[valid_df["legal_sectors_list"].apply(lambda x: sector in x)]
        prio_df = sector_df[sector_df["promulgation_year"].isin(TARGET_YEARS)].sort_values(by="promulgation_year", ascending=False)
        other_df = sector_df[~sector_df["promulgation_year"].isin(TARGET_YEARS)].sort_values(by="promulgation_year", ascending=False)
        
        combined_df = pd.concat([prio_df, other_df])
        selected_docs.extend([(sector, row) for _, row in combined_df.iterrows()])
        
    print("Initialized chunker...")
    chunker = AdvancedLegalChunker()
    encoder = LocalBGEHybridEncoder(DENSE_MODEL_NAME, DEVICE)
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))}
    )
    
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    sector_chunk_count = {s: 0 for s in TARGET_SECTORS}
    used_doc_ids = set()
    
    total_chunks = 0
    
    print("\n--- BEGIN PROCESSING ---")
    for target_sector, doc_row in selected_docs:
        if sector_chunk_count[target_sector] >= CHUNKS_PER_SECTOR:
            continue
            
        doc_id = str(doc_row["id"])
        if doc_id in used_doc_ids:
            # We don't process it but we count it if we intend to reuse
            pass
            
        doc_content = content_map.get(doc_id, "")
        if not doc_content: continue
        
        doc_meta, chunks = chunker.process_document(doc_content, doc_row.to_dict())
        if not chunks: continue
        
        used_doc_ids.add(doc_id)
        sector_chunk_count[target_sector] += len(chunks)
        total_chunks += len(chunks)
        
        print(f"[{target_sector}] {doc_meta['document_number']} ({doc_meta['promulgation_date']}) - {len(chunks)} chunks - Progress: {sector_chunk_count[target_sector]}/{CHUNKS_PER_SECTOR}")
        
        # 1. PUSH TO QDRANT
        texts = [c["text_to_embed"] for c in chunks]
        dense, sparse = encoder.encode_hybrid(texts)
        
        points = []
        for i, ch in enumerate(chunks):
            # Qdrant node IDs are integers or UUIDs. We use uuid derived from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ch["chunk_id"]))
            full_meta = {**doc_meta, **ch["metadata"]}
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": dense[i], "sparse": sparse[i]},
                    payload=full_meta
                )
            )
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        
        # 2. PUSH TO NEO4J
        build_neo4j(neo4j_driver, doc_meta, chunks)
        
    print(f"--- DONE! Processed {len(used_doc_ids)} docs into {total_chunks} chunks. ---")
    neo4j_driver.close()

if __name__ == '__main__':
    main()
