"""
AdvancedLegalChunker — Orchestrator mỏng, điều phối pipeline tách chunk.

Tái cấu trúc theo kiến trúc module hóa:
  core.py          ← file này   (Orchestrator)
  ├── doc_meta.py  ← Bóc căn cứ pháp lý, ngày hiệu lực
  ├── toc.py       ← Trích xuất mục lục
  ├── payload.py   ← DocContext + build_article_chunk + build_table_chunk
  ├── fsm.py       ← Vòng lặp FSM chính
  ├── heuristics.py← entity/relation hint (gọi bên trong payload builders)
  └── metadata.py  ← Regex patterns + tiện ích chung

Public API giữ nguyên để tương thích với toàn bộ pipeline:
  chunker.process_document(content, metadata, global_doc_lookup, precomputed_rels, skip_llm)
"""
import datetime
import uuid
from typing import Any, Dict, List

from backend.ingestion.chunker import metadata as md
from backend.ingestion.chunker import toc as toc_mod
from backend.ingestion.chunker.payload import DocContext
from backend.ingestion.chunker.fsm import scan_document
from backend.ingestion.extractor import relations as rel


class AdvancedLegalChunker:
    """
    Trái tim của hệ thống tách nhỏ văn bản pháp luật.

    process_document() thực hiện 8 bước theo thứ tự:
      1. Dọn dẹp nội dung thô
      2. Bóc metadata cơ bản (doc_id, doc_number, title, issuing_auth...)
      3. Ngày có hiệu lực & Căn cứ pháp lý  (doc_meta)
      4. Lĩnh vực pháp luật
      5. Ontology relations (hoặc dùng precomputed_rels)
      6. Trạng thái văn bản (doc_status)
      7. Mục lục (TOC)                        (toc_mod)
      8. FSM scan → chunks với entity/relation hints  (fsm)
    """

    def __init__(self):
        pass

    def process_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        global_doc_lookup: dict = None,
        precomputed_rels: List[dict] = None,
        skip_llm: bool = False,
    ) -> List[Dict[str, Any]]:

        # ──────────────────────────────────────────
        # 1. Dọn dẹp nội dung thô
        # ──────────────────────────────────────────
        content = str(content or "").replace("\r\n", "\n").strip()
        lines   = content.splitlines()

        # ──────────────────────────────────────────
        # 2. Metadata cơ bản
        # ──────────────────────────────────────────
        doc_id = str(metadata.get("id") or uuid.uuid4())

        doc_number = str(metadata.get("document_number") or "")
        if not doc_number or doc_number == "N/A":
            preamble_text = "\n".join(lines[:100])
            doc_number = str(md.extract_doc_number(preamble_text) or "N/A")

        doc_title       = str(metadata.get("title") or "N/A")
        issuing_auth    = str(metadata.get("issuing_authority") or "N/A")
        legal_type_meta = str(metadata.get("legal_type") or "N/A")
        url_meta        = str(metadata.get("url") or "")

        signer_name, signer_id = md.parse_signer(metadata.get("signers", ""))

        raw_promul = str(
            metadata.get("issuance_date") or
            metadata.get("promulgation_date") or ""
        )
        promulgation_date = raw_promul[:10] if len(raw_promul) >= 10 else raw_promul

        year = md.extract_year(raw_promul)
        if not year:
            year = md.extract_year(doc_number) or "N/A"

        # ──────────────────────────────────────────
        # 3. Ngày có hiệu lực & Căn cứ pháp lý
        # ──────────────────────────────────────────
        eff_date = md.extract_effective_date(content, promulgation_date)
        raw_eff  = str(eff_date or metadata.get("effective_date") or promulgation_date)
        final_effective_date = raw_eff[:10] if len(raw_eff) >= 10 else raw_eff

        basis_refs = md.extract_legal_basis(content, skip_llm=skip_llm)

        # ──────────────────────────────────────────
        # 4. Lĩnh vực pháp luật
        # ──────────────────────────────────────────
        raw_sectors = (
            metadata.get("legal_sectors") or
            metadata.get("legal_sectors_list") or []
        )
        if isinstance(raw_sectors, str):
            sectors_list = [s.strip() for s in raw_sectors.split(",") if s.strip()]
        elif isinstance(raw_sectors, list):
            sectors_list = [str(s).strip() for s in raw_sectors if str(s).strip()]
        else:
            sectors_list = []

        sectors_str = ", ".join(sectors_list) if sectors_list else "Chung"

        # ──────────────────────────────────────────
        # 5. Ontology relations
        # ──────────────────────────────────────────
        if precomputed_rels is not None:
            ontology_rels = precomputed_rels
            entities_side = {"entities": {}, "node_relations": []}
        else:
            ontology_rels, entities_side = rel.extract_ontology_relationships(
                content, doc_number, global_doc_lookup, skip_llm=skip_llm
            )

        # ──────────────────────────────────────────
        # 6. Trạng thái văn bản (doc_status)
        # ──────────────────────────────────────────
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        if (
            final_effective_date and
            len(final_effective_date) == 10 and
            final_effective_date > today_str
        ):
            doc_status = "Chưa có hiệu lực"
        else:
            doc_status = "Đang có hiệu lực"

        raw_doc_status = metadata.get("doc_status")
        doc_status = md.normalize_doc_status(
            raw_doc_status if raw_doc_status else doc_status
        )

        # ──────────────────────────────────────────
        # 7. Mục lục (TOC)
        # ──────────────────────────────────────────
        document_toc = toc_mod.extract_toc(lines)

        # ──────────────────────────────────────────
        # 8. Xây dựng DocContext & chạy FSM scan
        # ──────────────────────────────────────────
        ctx = DocContext(
            doc_id               = doc_id,
            doc_number           = doc_number,
            doc_title            = doc_title,
            year                 = year,
            sectors_list         = sectors_list,
            sectors_str          = sectors_str,
            doc_status           = doc_status,
            issuing_auth         = issuing_auth,
            signer_name          = signer_name,
            signer_id            = signer_id,
            promulgation_date    = promulgation_date,
            final_effective_date = final_effective_date,
            legal_type_meta      = legal_type_meta,
            url_meta             = url_meta,
            basis_refs           = basis_refs,
            document_toc         = document_toc,
            ontology_rels        = ontology_rels,
            entities             = entities_side.get("entities", {}),
            node_relations       = entities_side.get("node_relations", []),
        )

        # Một lần quét FSM duy nhất:
        # - Sinh ra tất cả chunks
        # - Gắn nhãn has_potential_entities & has_potential_relations inline
        return scan_document(lines, ctx)


# Singleton dùng chung toàn pipeline
chunker = AdvancedLegalChunker()
