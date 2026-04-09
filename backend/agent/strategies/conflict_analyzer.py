from typing import Dict, Any, List
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils_conflict_analyzer import (
    extract_claims_from_text,
    hyde_retrieve,
    cross_encoder_prune,
    cross_encoder_prune_with_scores,
    judge_claim,
    review_claim
)

class ConflictAnalyzerStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """
        Extractor & Queuer: 
        1. Lần đầu: Lấy file, trích claims Deontic, cho vào Queue (pending_tasks).
        2. Mỗi vòng lặp: Rút 2 claims từ Queue tạo thành mẻ Batch.
        """
        pending = state.get("pending_tasks")
        metadata = state.get("metadata_filters", {}) # Dùng tạm chứa org_metadata

        # --- INIT PHASE (Lần đầu chạy flow) ---
        if pending is None:
            file_chunks = state.get("file_chunks", [])
            query = state.get("condensed_query", state["query"])
            
            from backend.agent.utils_conflict_analyzer import route_conflict_intent
            
            if not file_chunks:
                intent = "NO_FILE"
                print(f"    ⚖️ [Conflict Analyzer] Flow 1: No file, extracting claim from query.")
            else:
                intent = route_conflict_intent(query, llm_preset=state.get("llm_preset"))
                print(f"    ⚖️ [Conflict Analyzer] Flow: {intent}")
                
                # Optimize: Rerank file chunks based on query before extraction
                from backend.agent.utils_conflict_analyzer import get_pruner
                import numpy as np
                model = get_pruner()
                if model and query:
                    q_emb = model.encode(query, show_progress_bar=False)
                    c_texts = [f.get("text_to_embed", f.get("unit_text", "")) for f in file_chunks]
                    c_emb = model.encode(c_texts, show_progress_bar=False)
                    sims = np.dot(c_emb, q_emb) / (np.linalg.norm(c_emb, axis=1) * np.linalg.norm(q_emb) + 1e-10)
                    scored = list(zip(file_chunks, sims))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    file_chunks = [item[0] for item in scored]
                
            metadata["conflict_intent"] = intent
            all_claims = []
            
            if intent == "NO_FILE":
                # Workflow 1: Single query, no file
                ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                if not ie_data.get("claims"):
                    all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                else:
                    for claim in ie_data.get("claims", []):
                        if claim.get("raw_text"): all_claims.append(claim)
                        
            elif intent == "VS_FILE" or intent == "VS_DB":
                if intent == "VS_FILE":
                    # Workflow 2: Extract claim from query, then search OVER file_chunks later
                    ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                    if not ie_data.get("claims"):
                        all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                    else:
                        for claim in ie_data.get("claims", []):
                            if claim.get("raw_text"): all_claims.append(claim)
                    
                    # Store file_chunks into state for later use
                    metadata["file_sources"] = file_chunks
                else:
                    # Workflow 3: Full file extraction (Original logic)
                    sample_chunks = file_chunks[:3]
                    print(f"    ⚖️ [Conflict Analyzer] Extracting policies from {len(sample_chunks)} file chunks...")
                    for c in sample_chunks:
                        chunk_text = c.get("text_to_embed", c.get("unit_text", ""))
                        ie_data = extract_claims_from_text(chunk_text, llm_preset=state.get("llm_preset"))
                        
                        if not metadata.get("co_quan_ban_hanh") and ie_data.get("metadata"):
                            metadata.update(ie_data.get("metadata"))
                            
                        for claim in ie_data.get("claims", []):
                            if claim.get("raw_text"):
                                all_claims.append(claim)
                        
            pending = all_claims
            print(f"       ✅ Khai thác thành công {len(pending)} mệnh đề (Claims). Bắt đầu Loop Batching.")
            
        # --- BATCH DISPATCH PHASE ---
        # Rút tối đa 2 claims mỗi batch để xử lý (Ngăn Rate Limit)
        # FREE Tier: Thêm delay giữa các mẻ xử lý nặng (70B)
        if pending is not None and len(all_claims if 'all_claims' in locals() else []) == 0:
            import asyncio
            print(f"       ⏱️ [Free Tier] Sleeping 20s to refill Groq TPM/RPM quota...")
            time.sleep(20) # Sử dụng sync sleep vì node_timer/graph có thể đang chạy sync wrapper
            
        batch_size = 2
        current_batch = pending[:batch_size]
        remaining = pending[batch_size:]
        
        # 'rewritten_queries' list now acts as the current batch container.
        return {
            "rewritten_queries": current_batch,
            "metadata_filters": metadata,  # Store document metadata
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
            # Flow 2: Use file_chunks instead of Qdrant
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

        # Flow 1 & 3: Qdrant Search
        use_rerank = state.get("use_rerank", False)
        for claim in current_batch:
            hits = hyde_retrieve(claim, use_rerank=use_rerank, llm_preset=state.get("llm_preset"))
            batch_hits.append({
                "claim": claim,
                "raw_qdrant_hits": hits
            })
            
        return {"raw_hits": batch_hits}

    def resolve_references(self, state: AgentState) -> Dict[str, Any]:
        """Truy xuất bổ sung Phụ lục cho từng Claim nếu được dẫn chiếu."""
        from backend.agent.utils_legal_qa import resolve_recursive_references
        batch_hits = state.get("raw_hits", [])
        if not batch_hits:
            return {"recursive_hits": []}
            
        resolved_batch = []
        for item in batch_hits:
            primary_hits = item["raw_qdrant_hits"]
            # To avoid affecting 'raw_hits' structure, we just compute recursive ones
            full_list = resolve_recursive_references(primary_hits)
            recursive_only = [h for h in full_list if h not in primary_hits]
            
            # Chúng ta gộp recursive_only vào raw_qdrant_hits của item này luôn để các bước sau (grade/judge) thấy được
            item["raw_qdrant_hits"] = full_list
            resolved_batch.append(item)
            
        return {"raw_hits": resolved_batch}

    def grade(self, state: AgentState) -> Dict[str, Any]:
        """
        Cross-Encoder Pruner:
        Sử dụng Local Model "all-MiniLM-L6-v2" trên CPU để giữ Top 3 Hits liên quan thực sự.
        """
        batch_hits = state.get("raw_hits", [])
        if not batch_hits:
            return {"filtered_context": []}
            
        pruned_batch = []
        is_best_effort = state.get("is_best_effort", False)
        for item in batch_hits:
            claim_obj = item["claim"]
            claim_text = claim_obj.get("raw_text", f"{claim_obj.get('chu_the')} {claim_obj.get('hanh_vi')}")
            
            # Prune to TOP 3
            hits = item["raw_qdrant_hits"]
            scored_hits = cross_encoder_prune_with_scores(claim_text, hits, top_k=3)
            top_3_hits = [h for h, s in scored_hits]
            max_score = scored_hits[0][1] if scored_hits else 0
            
            # Nếu score cao nhất < 0.2, coi như là Best-effort (không có luật trực tiếp)
            if max_score < 0.2 and top_3_hits:
                is_best_effort = True
            
            pruned_batch.append({
                "claim": claim_obj,
                "claim_text": claim_text,
                "top_hits": top_3_hits
            })
            
        # Override field with pruned list
        return {
            "filtered_context": pruned_batch,
            "is_best_effort": is_best_effort
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Judge Agent:
        Xử lý từng Claim trong chế độ Sequential. Đưa ra phán quyết (Label).
        """
        pruned_batch = state.get("filtered_context", [])
        metadata = state.get("metadata_filters", {})
        
        if not pruned_batch:
            return {"draft_response": ""}
            
        judge_results = []
        history_msgs = state.get("history", [])[-6:]
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
        
        for item in pruned_batch:
            dec = judge_claim(item["claim"], item["top_hits"], metadata, history_str=history_str, llm_preset=state.get("llm_preset"))
            judge_results.append({
                "claim_text": item["claim_text"],
                "hits": item["top_hits"],
                "judge_dec": dec
            })
            
        # Tạm chứa vào draft (hoặc pass qua field cho reflect xử tiếp)
        return {"draft_response": judge_results}

    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        Reviewer & Loop Controller:
        1. Gọi LLM check chống ảo giác.
        2. Nếu Queue (pending_tasks) còn -> Lệnh LangGraph quay lại understand.
        3. Nếu Queue hết -> Tổng hợp Markdown Report + References.
        """
        judge_results = state.get("draft_response", [])
        completed = state.get("completed_results", [])
        
        # 1. Review
        if isinstance(judge_results, list):
            for item in judge_results:
                final_dec = review_claim(item["claim_text"], item["judge_dec"], item["hits"], llm_preset=state.get("llm_preset"))
                completed.append({
                    "claim": item["claim_text"],
                    "label": final_dec.get("label", "Neutral"),
                    "reference_law": final_dec.get("reference_law", "N/A"),
                    "conflict_reasoning": final_dec.get("reasoning", ""),
                    "proposed_db_update": final_dec.get("proposed_db_update"),
                    "hits": item["hits"]
                })
                
        pending = state.get("pending_tasks", [])
        
        # 2. Loop Controller
        if pending:
            return {
                "completed_results": completed,
                "pass_flag": False  # Bắn cờ mồi để Graph quay đầu (router_reflect)
            }
            
        # 3. Kết thúc -> Build Master Report
        md_table = ""
        if state.get("is_best_effort"):
             md_table += "> [!NOTE]\n> **Thông báo:** Hệ thống không tìm thấy quy định trực tiếp, tuy nhiên dựa trên nội dung liên quan nhất tìm thấy, tôi xin cung cấp thông tin tham khảo như sau:\n\n"
             
        md_table += "### ⚠️ Kết Quả Phân Tích Xung Đột Pháp Lý\n\n"
        md_table += "| Mệnh đề Nội quy (Claim) | Phán quyết | Căn cứ Pháp lý | Giải thích chi tiết |\n"
        md_table += "| :--- | :---: | :--- | :--- |\n"
        
        references = []
        seen = set()
        
        has_db_update_proposal = False
        db_updates = []
        
        for r in completed:
            label_str = str(r['label']).lower()
            icon = "❌" if "contradiction" in label_str else ("✅" if "entailment" in label_str else "⚪")
            
            # Đảm bảo format "Căn cứ..." theo yêu cầu số 4
            formatted_reasoning = r['conflict_reasoning']
            ref_docs = r['reference_law']
            if "Căn cứ" not in ref_docs and "N/A" not in ref_docs:
                 ref_docs = f"Căn cứ {ref_docs}"
                 
            # Cần dọn dẹp các ký tự xuống dòng và dấu pipe để không làm hỏng bảng Markdown
            clean_claim = str(r['claim']).replace('\n', ' ').replace('|', '&#124;').strip()
            clean_reason = str(formatted_reasoning).replace('\n', ' ').replace('|', '&#124;').strip()
            clean_refs = str(ref_docs).replace('\n', ' ').replace('|', '&#124;').strip()
            
            md_table += f"| {clean_claim} | {icon} **{r['label']}** | {clean_refs} | {clean_reason} |\n"
            
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
                        "document_number": h.get("document_number", ""),
                        "url": h.get("url", "")
                    })
                    seen.add(cid)
        
        # Sắp xếp references theo score giảm dần (Rerank order)
        references.sort(key=lambda x: x.get("score", 0), reverse=True)
                    
        # Nếu có đề xuất update DB, trả kèm cờ JSON ẩn hoặc markdown block để frontend xử lý
        if has_db_update_proposal:
            import json
            old_nums = list(set([u.get("old_document_number") for u in db_updates if u.get("old_document_number")]))
            new_num = db_updates[0].get("new_document_number") if db_updates else None
            
            # Lấy thông tin file tài liệu hiện tại từ state
            file_path = state.get("file_path")
            new_file_id = None
            new_filename = None
            if file_path:
                new_filename = os.path.basename(file_path)
                new_file_id = os.path.splitext(new_filename)[0]

            update_payload = {
                "document_numbers_to_disable": old_nums,
                "new_file_id": new_file_id,
                "new_filename": new_filename
            }
            md_table += f"\n\n<!-- DB_UPDATE_PROPOSAL: {json.dumps(update_payload)} -->\n"
            md_table += f"> **⚠️ Đề Xuất Cập Nhật Cơ Sở Dữ Liệu:** Hệ thống phát hiện văn bản mới ({new_num}) có tính chất thay thế các văn bản cũ ({', '.join(old_nums)}). Vui lòng xác nhận để đồng bộ cập nhật dữ liệu Qdrant."
                    
        return {
            "completed_results": completed,
            "final_response": md_table,
            "references": references,
            "pass_flag": True  # Chốt chặn luồng đồ thị
        }
