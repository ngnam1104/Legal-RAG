from typing import Dict, Any, List
import os
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils.utils_conflict_analyzer import (
    extract_claims_from_text,
    hyde_retrieve,
    judge_claim
)
from backend.agent.utils.utils_legal_qa import filter_cited_references
from backend.retrieval.chunker.metadata import extract_doc_number

class ConflictAnalyzerStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """
        Extractor & Queuer: 
        1. Lần đầu: Lấy file, trích claims Deontic, cho vào Queue (pending_tasks).
        2. Mỗi vòng lặp: Rút 2 claims từ Queue tạo thành mẻ Batch.
        """
        pending = state.get("pending_tasks")
        metadata = state.get("metadata_filters", {})

        if pending is None:
            file_chunks = state.get("file_chunks", [])
            query = state.get("condensed_query", state["query"])
            
            from backend.agent.utils.utils_conflict_analyzer import route_conflict_intent
            
            if not file_chunks:
                intent = "NO_FILE"
            else:
                intent = route_conflict_intent(query, llm_preset=state.get("llm_preset"))
                
                from backend.retrieval.reranker import reranker as api_reranker
                if query and file_chunks:
                    try:
                        file_chunks = api_reranker.rerank_candidates(query=query, candidates=file_chunks, top_k=len(file_chunks))
                    except Exception as e:
                        pass
                
            metadata["conflict_intent"] = intent
            all_claims = []
            
            if intent == "NO_FILE":
                # Trích xuất doc_numbers từ câu hỏi user (nếu nhắc tên cụ thể)
                import re
                doc_nums_in_query = re.findall(r'\d+/\d{4}/[A-Za-zĐđ\-]+', query)
                if doc_nums_in_query:
                    metadata["compare_doc_numbers"] = doc_nums_in_query[:2]
                
                ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                if not ie_data.get("claims"):
                    all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                else:
                    for claim in ie_data.get("claims", []):
                        if claim.get("raw_text"): all_claims.append(claim)
                        
            elif intent == "VS_FILE" or intent == "VS_DB":
                if intent == "VS_FILE":
                    ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                    if not ie_data.get("claims"):
                        all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                    else:
                        for claim in ie_data.get("claims", []):
                            if claim.get("raw_text"): all_claims.append(claim)
                    metadata["file_sources"] = file_chunks
                else:
                    sample_chunks = file_chunks[:3]
                    for c in sample_chunks:
                        chunk_text = c.get("text_to_embed", c.get("unit_text", ""))
                        ie_data = extract_claims_from_text(chunk_text, llm_preset=state.get("llm_preset"))
                        
                        if not metadata.get("co_quan_ban_hanh") and ie_data.get("metadata"):
                            metadata.update(ie_data.get("metadata"))
                            
                        for claim in ie_data.get("claims", []):
                            if claim.get("raw_text"):
                                all_claims.append(claim)
                        
            pending = all_claims
            
        rewritten = state.get("rewritten_queries")
        rcount = state.get("retry_count", 0)
        sufficient = state.get("is_sufficient")
        
        is_retry = (rcount > 0) and (sufficient is not True)
        
        if is_retry and rewritten:
            current_batch = rewritten
            remaining = pending if pending is not None else []
        else:
            batch_size = 2
            if pending is None:
                current_batch = []
                remaining = []
            else:
                current_batch = pending[:batch_size]
                remaining = pending[batch_size:]

        return {
            "rewritten_queries": current_batch,
            "metadata_filters": metadata,
            "pending_tasks": remaining
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        Conflict Hunting:
        Chạy vòng lặp (For) dùng HyDE và truy vấn Qdrant cho mỗi claim trong Batch.
        """
        current_batch = state.get("rewritten_queries", [])
        if not current_batch:
            return {"raw_hits": []}
            
        intent = state.get("metadata_filters", {}).get("conflict_intent", "VS_DB")
        batch_hits = []
        
        if intent == "VS_FILE":
            file_sources = state.get("metadata_filters", {}).get("file_sources", [])
            file_hits = []
            for idx, c in enumerate(file_sources):
                file_hits.append({
                    "id": f"file_{idx}",
                    "score": 1.0,
                    "text": c.get("text_to_embed", c.get("unit_text", "")),
                    "title": "Tài liệu đính kèm",
                    "article_ref": f"Phần {idx+1}",
                    "document_number": "FILE",
                    "is_active": True
                })
            for claim in current_batch:
                batch_hits.append({
                    "claim": claim,
                    "raw_qdrant_hits": file_hits
                })
            return {"raw_hits": batch_hits}

        use_rerank = state.get("use_rerank", True)
        
        compare_doc_numbers = state.get("metadata_filters", {}).get("compare_doc_numbers", [])
        if compare_doc_numbers:
            graph_hits = []
            try:
                from backend.retrieval.graph_db import fetch_full_document_articles
                for doc_num in compare_doc_numbers:
                    articles = fetch_full_document_articles(doc_num)
                    for a in articles:
                        if not a.get("article_text") and not a.get("clauses"):
                            continue
                        text_content = a.get("article_text", "")
                        for cl in a.get("clauses", []):
                            text_content += f"\n{cl.get('name')}: {cl.get('text')}"
                        
                        fake_hit = {
                            "id": f"compare_{doc_num}_{a.get('article_ref')}",
                            "score": 1.0,
                            "document_number": a.get("document_number"),
                            "title": a.get("title"),
                            "text": text_content,
                            "article_ref": a.get("article_ref"),
                            "is_active": True,
                            "chunk_id": f"compare_{doc_num}_{a.get('article_ref')}"
                        }
                        graph_hits.append(fake_hit)
            except Exception as e:
                print(f"Error fetching compare docs: {e}")
                pass
                
            if graph_hits:
                for claim in current_batch:
                    batch_hits.append({
                        "claim": claim,
                        "raw_qdrant_hits": graph_hits
                    })
            else:
                for claim in current_batch:
                    hits = hyde_retrieve(claim, use_rerank=use_rerank, llm_preset=state.get("llm_preset"))
                    batch_hits.append({
                        "claim": claim,
                        "raw_qdrant_hits": hits
                    })
        else:
            for claim in current_batch:
                hits = hyde_retrieve(claim, use_rerank=use_rerank, llm_preset=state.get("llm_preset"))
                batch_hits.append({
                    "claim": claim,
                    "raw_qdrant_hits": hits
                })
        
        graph_context = state.get("graph_context", {})
        graph_context.update({"time_travel": [], "lateral_docs": []})
        all_chunk_ids = []
        for item in batch_hits:
            chunk_ids = [h.get("chunk_id", "") for h in item["raw_qdrant_hits"] if h.get("chunk_id")]
            if chunk_ids:
                all_chunk_ids.extend(chunk_ids)
                try:
                    from backend.retrieval.graph_db import conflict_time_travel
                    tt_results = conflict_time_travel(chunk_ids)
                    if tt_results:
                        graph_context["time_travel"].extend(tt_results)
                except Exception as e:
                    pass
                    
        if all_chunk_ids:
            try:
                from backend.retrieval.graph_db import lateral_expand
                graph_context["lateral_docs"] = lateral_expand(list(set(all_chunk_ids)))
            except Exception:
                pass
            
        return {"raw_hits": batch_hits, "graph_context": graph_context}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Judge Agent:
        Xử lý từng Claim. Đưa thẳng bối cảnh Time-Travel vào Prompt.
        """
        batch_hits = state.get("raw_hits", [])
        if not batch_hits:
            no_results_msg = "Hệ thống không tìm thấy văn bản pháp luật liên quan nào trong CSDL để đối chiếu với các mệnh đề này."
            return {"final_response": no_results_msg, "conflict_draft": []}

        # Bypass Grade Logic: Prune to top 3 manually instead of running cross-encoder node
        pruned_batch = []
        is_best_effort = state.get("is_best_effort", False)
        for item in batch_hits:
            claim_obj = item["claim"]
            claim_text = claim_obj.get("raw_text", f"{claim_obj.get('chu_the')} {claim_obj.get('hanh_vi')}")
            top_3_hits = item["raw_qdrant_hits"][:3]
            pruned_batch.append({
                "claim": claim_obj,
                "claim_text": claim_text,
                "top_hits": top_3_hits
            })
            
        metadata = state.get("metadata_filters", {})
        graph_ctx = state.get("graph_context", {})
        time_travel_data = graph_ctx.get("time_travel", []) if graph_ctx else []
            
        judge_results = []
        history_msgs = state.get("history", [])[-6:]
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
        
        tt_map = {}
        for tt in time_travel_data:
            key = tt.get("original_doc_number", "")
            if key and tt.get("amending_doc_number"):
                if key not in tt_map:
                    tt_map[key] = []
                tt_map[key].append(tt)
        
        for item in pruned_batch:
            claim_obj = item["claim"]
            top_hits = item["top_hits"]
            
            deontic_context = ""
            for h in top_hits:
                doc_num = h.get("document_number", "")
                if doc_num in tt_map:
                    for tt in tt_map[doc_num]:
                        deontic_context += (
                            f"\\n[CẢNH BÁO HIỆU LỰC] Văn bản {doc_num} đã bị tác động bởi {tt.get('amending_doc_number', 'N/A')} "
                            f"(Quan hệ: {tt.get('relation_type', 'N/A')}).\\n"
                        )
                        if tt.get("old_text"):
                            deontic_context += f"Nội dung cũ: {tt['old_text']}\\n"
                        if tt.get("new_text"):
                            deontic_context += f"Nội dung hiện hành: {tt['new_text']}\\n"
            
            admin_meta = state.get("graph_context", {}).get("admin_metadata", "")
            if admin_meta:
                 deontic_context += f"\\n[THÔNG TIN HÀNH CHÍNH]: {admin_meta}\\n"
                 
            if deontic_context:
                metadata["deontic_context"] = deontic_context
            
            dec = judge_claim(item["claim"], item["top_hits"], metadata, history_str=history_str, llm_preset=state.get("llm_preset"))
            judge_results.append({
                "claim": item["claim_text"],
                "label": dec.get("label", "Neutral"),
                "reference_law": dec.get("reference_law", "N/A"),
                "conflict_reasoning": dec.get("reasoning", ""),
                "proposed_db_update": dec.get("proposed_db_update"),
                "hits": item["top_hits"]
            })
            
        completed = state.get("completed_results", []) + judge_results
        pending = state.get("pending_tasks", [])
        
        if pending:
            return {"completed_results": completed}
            
        # --- Build Master Report ---
        query = state.get("query", "").lower()
        wh_words = ["làm thế nào", "cơ quan nào", "có văn bản nào", "ai ", "khi nào", "tại sao", "ở đâu", "văn bản nào", "thế nào"]
        is_wh_query = any(word in query for word in wh_words)

        if len(completed) == 1 and is_wh_query:
            r = completed[0]
            label_str = str(r['label']).lower()
            icon = "❌" if "contradiction" in label_str else ("✅" if "entailment" in label_str else "⚪")
            
            final_ans = f"### 💡 Kết Luận Phân Tích\\n\\n"
            final_ans += f"**{icon} Phán quyết:** {r['label']}\\n\\n"
            final_ans += f"**Giải thích chi tiết:** {r['conflict_reasoning']}\\n"
            
            md_table = final_ans
        else:
            md_table = ""
            if is_best_effort:
                 md_table += "> [!NOTE]\\n> **Thông báo:** Hệ thống không tìm thấy quy định trực tiếp, tuy nhiên dựa trên nội dung liên quan nhất tìm thấy, tôi xin cung cấp thông tin tham khảo như sau:\\n\\n"
                 
            md_table += "### ⚠️ Kết Quả Phân Tích Xung Đột Pháp Lý\\n\\n"
            md_table += "| Mệnh đề Nội quy (Claim) | Phán quyết | Giải thích chi tiết |\\n"
            md_table += "| :--- | :---: | :--- |\\n"
            
            for r in completed:
                label_str = str(r['label']).lower()
                icon = "❌" if "contradiction" in label_str else ("✅" if "entailment" in label_str else "⚪")
                
                clean_claim = str(r['claim']).replace('\\n', ' ').replace('|', '&#124;').strip()
                clean_reason = str(r['conflict_reasoning']).replace('\\n', ' ').replace('|', '&#124;').strip()
                
                md_table += f"| {clean_claim} | {icon} **{r['label']}** | {clean_reason} |\\n"
        
        references = []
        seen = set()
        
        has_db_update_proposal = False
        db_updates = []
        
        for r in completed:
            if r.get("proposed_db_update") and r["proposed_db_update"].get("is_needed"):
                has_db_update_proposal = True
                db_updates.append(r["proposed_db_update"])
            
            for h in r.get("hits", []):
                cid = h.get("chunk_id")
                if cid and cid not in seen:
                    references.append({
                        "title": h.get("title", ""),
                        "article": h.get("article_ref", h.get("document_number", "")),
                        "score": h.get("score", 0),
                        "chunk_id": cid,
                        "text_preview": h.get("text", ""),
                        "document_number": h.get("document_number", ""),
                        "url": h.get("url", "")
                    })
                    seen.add(cid)
        
        references.sort(key=lambda x: x.get("score", 0), reverse=True)
        cited_refs = filter_cited_references(md_table, references)
                    
        if has_db_update_proposal:
            import json
            # Làm sạch danh sách số hiệu văn bản cũ (chỉ lấy số hiệu, bỏ rác)
            raw_old_nums = [u.get("old_document_number") for u in db_updates if u.get("old_document_number")]
            old_nums = list(set([extract_doc_number(n) for n in raw_old_nums if extract_doc_number(n)]))
            
            # Ưu tiên lấy new_num từ metadata của file tải lên nếu AI trả về N/A
            ai_new_num = db_updates[0].get("new_document_number") if db_updates else None
            metadata = state.get("metadata_filters", {})
            file_doc_num = metadata.get("document_number")
            
            new_num = ai_new_num
            if (not new_num or new_num.upper() == "N/A") and file_doc_num:
                new_num = file_doc_num
            
            file_path = state.get("file_path")
            new_file_id = None
            new_filename = None
            if file_path:
                import os
                new_filename = os.path.basename(file_path)
                new_file_id = os.path.splitext(new_filename)[0]

            update_payload = {
                "document_numbers_to_disable": old_nums,
                "new_file_id": new_file_id,
                "new_filename": new_filename
            }
            md_table += f"\\n\\n<!-- DB_UPDATE_PROPOSAL: {json.dumps(update_payload)} -->\\n"
            
            # Hiển thị thông tin sạch sẽ cho người dùng
            new_num_display = new_num if new_num and new_num.upper() != "N/A" else "mới tải lên"
            old_nums_display = ", ".join(old_nums) if old_nums else "các văn bản cũ"
            
            md_table += f"> **⚠️ Đề Xuất Cập Nhật Cơ Sở Dữ Liệu:** Hệ thống phát hiện văn bản {new_num_display} có tính chất thay thế {old_nums_display}. Vui lòng xác nhận để đồng bộ cập nhật dữ liệu Qdrant."
                    
        return {
            "completed_results": completed,
            "final_response": md_table,
            "references": cited_refs
        }
