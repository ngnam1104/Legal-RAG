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
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Understand")
        
        pending = state.get("pending_tasks")
        metadata = state.get("metadata_filters", {})

        if pending is None:
            file_chunks = state.get("file_chunks", [])
            query = state.get("condensed_query", state["query"])
            
            from backend.agent.utils.utils_conflict_analyzer import route_conflict_intent
            
            if not file_chunks:
                intent = "NO_FILE"
            else:
                with timer.step("Conflict_Intent"):
                    intent = route_conflict_intent(query, llm_preset=state.get("llm_preset"))
                
                from backend.retrieval.reranker import reranker as api_reranker
                if query and file_chunks:
                    try:
                        with timer.step("Rerank_Files"):
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
                
                with timer.step("ClaimExtraction"):
                    ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                    if not ie_data.get("claims"):
                        all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                    else:
                        for claim in ie_data.get("claims", []):
                            if claim.get("raw_text"): all_claims.append(claim)
                        
            elif intent == "VS_FILE" or intent == "VS_DB":
                if intent == "VS_FILE":
                    with timer.step("ClaimExtraction"):
                        ie_data = extract_claims_from_text(query, llm_preset=state.get("llm_preset"))
                        if not ie_data.get("claims"):
                            all_claims.append({"chu_the": "Người dùng", "hanh_vi": query, "dieu_kien": "", "he_qua": "", "raw_text": query})
                        else:
                            for claim in ie_data.get("claims", []):
                                if claim.get("raw_text"): all_claims.append(claim)
                    metadata["file_sources"] = file_chunks
                else:
                    sample_chunks = file_chunks[:3]
                    with timer.step("ClaimExtraction_File"):
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
            "pending_tasks": remaining,
            "metrics": timer.results()
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        Conflict Hunting:
        Chạy vòng lặp (For) dùng HyDE và truy vấn Qdrant cho mỗi claim trong Batch.
        """
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Retrieve")
        
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
            with timer.step("Session_Search"):
                for claim in current_batch:
                    batch_hits.append({
                        "claim": claim,
                        "raw_qdrant_hits": file_hits
                    })
            return {"raw_hits": batch_hits, "metrics": timer.results()}

        use_rerank = state.get("use_rerank", True)
        
        compare_doc_numbers = state.get("metadata_filters", {}).get("compare_doc_numbers", [])
        if compare_doc_numbers:
            graph_hits = []
            try:
                from backend.retrieval.graph_db import fetch_full_document_articles
                with timer.step("Graph_Compare_Fetch"):
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
                
            for claim in current_batch:
                with timer.step("HyDE_Retrieve"):
                    hits = hyde_retrieve(claim, use_rerank=use_rerank, llm_preset=state.get("llm_preset"))
                if graph_hits:
                    # Combine graph_hits and hits, preferring graph_hits if same document
                    seen_docs = {h.get("document_number") for h in graph_hits if h.get("document_number")}
                    filtered_hits = [h for h in hits if h.get("document_number") not in seen_docs]
                    combined = graph_hits + filtered_hits
                else:
                    combined = hits
                batch_hits.append({
                    "claim": claim,
                    "raw_qdrant_hits": combined
                })
        else:
            for claim in current_batch:
                with timer.step("HyDE_Retrieve"):
                    hits = hyde_retrieve(claim, use_rerank=use_rerank, llm_preset=state.get("llm_preset"))
                batch_hits.append({
                    "claim": claim,
                    "raw_qdrant_hits": hits
                })
        
        graph_context = state.get("graph_context", {})
        graph_context.update({"time_travel": []})
        all_chunk_ids = []
        for item in batch_hits:
            chunk_ids = [h.get("chunk_id", "") for h in item["raw_qdrant_hits"] if h.get("chunk_id")]
            if chunk_ids:
                all_chunk_ids.extend(chunk_ids)
                try:
                    from backend.retrieval.graph_db import conflict_time_travel, detect_conflicting_documents, conflict_chain_detect
                    from backend.retrieval.ingestion import fetch_old_text_from_qdrant
                    
                    tt_results = conflict_time_travel(chunk_ids)
                    if tt_results:
                        for tt in tt_results:
                            # Fetch chunk text cho văn bản sửa đổi (new_text) nếu có
                            amending_doc = tt.get("amending_doc_number")
                            target_art = tt.get("target_article")
                            if amending_doc and target_art:
                                text_found = fetch_old_text_from_qdrant(amending_doc, target_art)
                                if text_found:
                                    tt["new_text"] = text_found
                            # Gán old_text = target_text từ Graph
                            tt["old_text"] = tt.get("target_text", "")
                        
                        graph_context["time_travel"].extend(tt_results)
                        
                    # Yêu cầu 3: Graph Conflict Detection (Traverse AMENDS/REPLACES)
                    for h in item["raw_qdrant_hits"]:
                        dnum = h.get("document_number")
                        aref = h.get("article_ref")
                        if dnum and aref:
                            graph_conflicts = detect_conflicting_documents(dnum, aref)
                            for gc in graph_conflicts:
                                amending_doc = gc.get("amending_doc")
                                target_art = gc.get("target_article")
                                new_text = ""
                                if amending_doc and target_art:
                                    new_text = fetch_old_text_from_qdrant(amending_doc, target_art)
                                    
                                # Chuyển format để tương thích với tt_results
                                graph_context["time_travel"].append({
                                    "original_doc_number": dnum,
                                    "original_title": h.get("title", ""),
                                    "amending_doc_number": amending_doc,
                                    "amending_doc_title": gc.get("title"),
                                    "relation_type": gc.get("relation_type"),
                                    "target_article": target_art,
                                    "context": gc.get("context"),
                                    "new_text": new_text
                                })
                    # 2-Hop Chain Detection: tìm chuỗi sửa đổi gián tiếp
                    for h in item["raw_qdrant_hits"]:
                        dnum = h.get("document_number")
                        aref = h.get("article_ref", "")
                        if dnum:
                            chains = conflict_chain_detect(dnum, aref)
                            for chain in chains:
                                if chain.get("depth", 1) > 1:  # Chỉ lấy chuỗi gián tiếp (depth > 1)
                                    graph_context["time_travel"].append({
                                        "original_doc_number": dnum,
                                        "original_title": h.get("title", ""),
                                        "amending_doc_number": chain["chain_doc"],
                                        "amending_doc_title": chain.get("chain_title", ""),
                                        "relation_type": " → ".join(chain.get("chain_types", [])),
                                        "target_article": ", ".join([a for a in (chain.get("chain_articles") or []) if a]),
                                        "context": "; ".join([c for c in (chain.get("chain_contexts") or []) if c])[:500],
                                        "is_indirect_chain": True,
                                        "chain_depth": chain.get("depth", 2)
                                    })
                                    
                except Exception as e:
                    print(f"Lỗi Graph Conflict Retrieve: {e}")
                    pass
            
        return {"raw_hits": batch_hits, "graph_context": graph_context, "metrics": timer.results()}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Judge Agent:
        Xử lý từng Claim. Đưa thẳng bối cảnh Time-Travel vào Prompt.
        """
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Generate")
        
        batch_hits = state.get("raw_hits", [])
        if not batch_hits:
            no_results_msg = "Hệ thống không tìm thấy văn bản pháp luật liên quan nào trong CSDL để đối chiếu với các mệnh đề này."
            return {"final_response": no_results_msg, "conflict_draft": [], "metrics": timer.results()}

        # Bypass Grade Logic: Prune to top 8 manually, boost target document
        pruned_batch = []
        is_best_effort = state.get("is_best_effort", False)
        
        with timer.step("BuildContext"):
            for item in batch_hits:
                claim_obj = item["claim"]
                claim_text = claim_obj.get("raw_text", f"{claim_obj.get('chu_the')} {claim_obj.get('hanh_vi')}")
                
                top_hits = item["raw_qdrant_hits"]
                top_hits = sorted(top_hits, key=lambda x: x.get("score", 0.0), reverse=True)[:20]
                
                pruned_batch.append({
                    "claim": claim_obj,
                    "claim_text": claim_text,
                    "top_hits": top_hits
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
        
        from backend.agent.utils.utils_conflict_analyzer import score_contradiction_by_embedding
        
        for item in pruned_batch:
            claim_obj = item["claim"]
            top_hits = item["top_hits"]
            deontic_context = ""
            for h in top_hits:
                doc_num = h.get("document_number", "")
                if doc_num in tt_map:
                    for tt in tt_map[doc_num]:
                        deontic_context += (
                            f"\n[CẢNH BÁO HIỆU LỰC] Văn bản {doc_num} đã bị tác động bởi {tt.get('amending_doc_number', 'N/A')} "
                            f"(Quan hệ: {tt.get('relation_type', 'N/A')}).\n"
                        )
                        if tt.get("old_text"):
                            deontic_context += f"Nội dung cũ: {tt['old_text']}\n"
                        if tt.get("new_text"):
                            deontic_context += f"Nội dung hiện hành: {tt['new_text']}\n"
                            
                # Semantic contradiction score computation
                hit_text = h.get("text", "")
                if hit_text and len(hit_text) > 20:
                    with timer.step("Semantic_Score"):
                        div_score = score_contradiction_by_embedding(item.get("claim_text", ""), hit_text)
                    if div_score > 0.6:
                         deontic_context += f"\n[CẢNH BÁO SEMANTIC]: Mức độ xung đột ngữ nghĩa (Divergence Score) với {doc_num} là {div_score:.2f} (rất cao).\n"
            
            admin_meta = state.get("graph_context", {}).get("admin_metadata", "")
            if admin_meta:
                 deontic_context += f"\n[THÔNG TIN HÀNH CHÍNH]: {admin_meta}\n"
                 
            if deontic_context:
                metadata["deontic_context"] = deontic_context
            
            with timer.step("LLM_Judge"):
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
            
        # --- Build Master Report: ALWAYS output BOTH prose + table ---
        md_table = ""
        if is_best_effort:
             md_table += "> [!NOTE]\n> **Thông báo:** Hệ thống không tìm thấy quy định trực tiếp, tuy nhiên dựa trên nội dung liên quan nhất tìm thấy, tôi xin cung cấp thông tin tham khảo như sau:\n\n"

        # PHẦN 1: Prose Reasoning (Giải thích tổng quan bằng văn bản)
        md_table += "### 💡 Kết Luận Phân Tích\n\n"
        for r in completed:
            label_str = str(r['label']).lower()
            icon = "❌" if "contradiction" in label_str else ("✅" if "entailment" in label_str else "⚪")
            clean_claim = str(r['claim']).replace('\n', ' ').strip()
            
            md_table += f"**{icon} Mệnh đề:** {clean_claim}\n\n"
            md_table += f"**Phán quyết:** {r['label']}\n\n"
            md_table += f"{r['conflict_reasoning']}\n\n"
            if r.get('reference_law') and r['reference_law'] != 'N/A':
                md_table += f"📌 *Căn cứ: {r['reference_law']}*\n\n"
            md_table += "---\n\n"

        # PHẦN 2: Bảng tổng hợp (Table)
        md_table += "### 📊 Bảng Tổng Hợp Kết Quả\n\n"
        md_table += "| STT | Mệnh đề (Claim) | Phán quyết | Căn cứ | Tóm tắt |\n"
        md_table += "| :---: | :--- | :---: | :--- | :--- |\n"
        
        for idx, r in enumerate(completed, 1):
            label_str = str(r['label']).lower()
            icon = "❌" if "contradiction" in label_str else ("✅" if "entailment" in label_str else "⚪")
            
            clean_claim = str(r['claim']).replace('\n', ' ').replace('|', '&#124;').strip()
            clean_reason = str(r['conflict_reasoning']).replace('\n', ' ').replace('|', '&#124;').strip()
            ref_law = str(r.get('reference_law', 'N/A')).replace('|', '&#124;').strip()
            # Cắt ngắn reasoning cho bảng, giữ đầy đủ ở phần prose
            short_reason = clean_reason[:200] + "..." if len(clean_reason) > 200 else clean_reason
            
            md_table += f"| {idx} | {clean_claim} | {icon} **{r['label']}** | {ref_law} | {short_reason} |\n"
        
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
        with timer.step("FilterRefs"):
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
            md_table += f"\n\n<!-- DB_UPDATE_PROPOSAL: {json.dumps(update_payload)} -->\n"
            
            # Hiển thị thông tin sạch sẽ cho người dùng
            new_num_display = new_num if new_num and new_num.upper() != "N/A" else "mới tải lên"
            old_nums_display = ", ".join(old_nums) if old_nums else "các văn bản cũ"
            
            md_table += f"> **⚠️ Đề Xuất Cập Nhật Cơ Sở Dữ Liệu:** Hệ thống phát hiện văn bản {new_num_display} có tính chất thay thế {old_nums_display}. Vui lòng xác nhận để đồng bộ cập nhật dữ liệu Qdrant."
                    
        combined_thinking = "\n\n---\n\n".join([r.get("thinking_content", "").strip() for r in completed if r.get("thinking_content", "").strip()])
        return {
            "completed_results": completed,
            "final_response": md_table,
            "thinking_content": combined_thinking,
            "references": cited_refs,
            "metrics": timer.results()
        }
