"use client";
import React, { useState, useRef } from "react";
import { Paperclip, ArrowUp, X, Square } from "lucide-react";
import TextareaAutosize from "react-textarea-autosize";
import { useChat } from "@/contexts/ChatContext";
import toast from "react-hot-toast";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export default function ChatInput() {
  const { 
    sendMessage, uploadFile, ingestFile, clearStagedFile, 
    isSending, isIngesting, stagedFile, 
    stopResponse, inputBuffer, setInputBuffer, 
    isPendingEdit, editingIndex, cancelEdit, activeMode 
  } = useChat();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleSend = () => {
    if (!inputBuffer.trim() || isSending) return;
    const query = inputBuffer;
    setInputBuffer(""); // clear instantly
    sendMessage(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Upload file to backend - Staged only
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['pdf', 'docx', 'doc'].includes(ext || '')) {
      toast.error("Chỉ hỗ trợ file PDF, DOCX, DOC");
      return;
    }

    const toastId = toast.loading(`Đang tải lên tạm thời ${file.name}...`);
    try {
      await uploadFile(file);
      toast.success(
        <div>
          <strong>Tải lên tạm thời thành công!</strong>
          <p className="text-xs mt-1 text-gray-500">Bạn có thể hỏi đáp ngay hoặc nhấn "Đưa lên DB" để lưu vĩnh viễn.</p>
        </div>,
        { id: toastId, duration: 5000 }
      );
    } catch (error: any) {
      toast.error(`Upload thất bại: ${error.message}`, { id: toastId });
    }
  };

  const handleIngest = async () => {
    if (!stagedFile) return;
    const toastId = toast.loading(`Đang kiểm tra và đưa ${stagedFile.name} lên DB...`);
    
    try {
      const data = await ingestFile();
      
      if (data.status === "duplicate") {
        toast.error(data.message, { id: toastId, duration: 6000 });
        return;
      }

      const taskId = data.task_id;
      // Polling function
      const pollTimer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/task-status/${taskId}`);
          if (res.ok) {
            const statusData = await res.json();
            if (statusData.status === "completed") {
              clearInterval(pollTimer);
              clearStagedFile();
              toast.success(
                <div>
                  <strong>{statusData.result.filename} đã được đưa lên DB!</strong><br/>
                  <p className="text-sm mt-1">{statusData.result.summary}</p>
                </div>, 
                { id: toastId, duration: 8000 }
              );
            } else if (statusData.status === "failed") {
              clearInterval(pollTimer);
              toast.error(`Lỗi nạp DB: ${statusData.error}`, { id: toastId });
            }
          }
        } catch (err) {
          console.error("Polling error", err);
        }
      }, 3000);

      // Stop polling after 3 minutes
      setTimeout(() => clearInterval(pollTimer), 180000);
    } catch (error: any) {
      toast.error(`Lỗi: ${error.message}`, { id: toastId });
    }
  };

  const isGeneralChat = activeMode === "GENERAL_CHAT";

  return (
    <div className="flex flex-col gap-2 w-full">
      {/* File Staging indicator if any */}
      {stagedFile && !isGeneralChat && (
        <div className="flex items-center justify-between px-4 py-2 bg-emerald-primary/10 border border-emerald-primary/30 rounded-xl mb-1 animate-in fade-in slide-in-from-bottom-2">
          <div className="flex items-center gap-2 overflow-hidden">
             <Paperclip size={14} className="text-emerald-accent" />
             <span className="text-xs font-bold text-emerald-accent truncate">{stagedFile.name} (Chưa nạp DB)</span>
          </div>
          <div className="flex gap-2">
             <button onClick={handleIngest} disabled={isIngesting} className="text-[10px] font-black uppercase tracking-widest text-emerald-base bg-emerald-accent px-3 py-1 rounded-md hover:brightness-110 disabled:opacity-50">
               {isIngesting ? "Đang nạp..." : "Nạp DB"}
             </button>
             <button onClick={clearStagedFile} className="text-text-disabled hover:text-red-400">
               <X size={14} />
             </button>
          </div>
        </div>
      )}

      <div className="flex items-end gap-2 bg-emerald-surface/90 backdrop-blur-xl rounded-[2rem] p-2 pr-4 pl-4 border border-emerald-primary/20 shadow-2xl focus-within:border-emerald-primary/60 focus-within:shadow-[0_0_25px_rgba(0,255,180,0.15)] transition-all group/input relative">
        
        {/* Pending Edit Indicator / Cancel Button */}
        {isPendingEdit && (
          <button 
            onClick={cancelEdit}
            className="p-3 text-red-400 hover:bg-red-900/20 rounded-full transition-colors mb-1"
            title="Hủy sửa"
            type="button"
          >
            <X size={20} />
          </button>
        )}

        {/* Hidden file input */}
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          className="hidden" 
          accept=".pdf,.doc,.docx"
        />

        {/* Upload File Button */}
        {!isGeneralChat && (
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isSending}
            className="p-3 text-text-disabled hover:text-emerald-accent hover:bg-emerald-primary/10 rounded-full transition-all mb-1 disabled:opacity-50"
            title="Đính kèm tài liệu"
            type="button"
          >
            <Paperclip size={20} />
          </button>
        )}

        {/* Input Textarea */}
        <TextareaAutosize
          value={inputBuffer}
          onChange={(e) => setInputBuffer(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isSending 
              ? "Đang xử lý..." 
              : editingIndex !== null
                ? "Đang sửa tin nhắn phía trên..."
                : isGeneralChat 
                  ? "Bắt đầu cuộc trò chuyện..." 
                  : "Tra cứu pháp luật hoặc phân tích tài liệu..."
          }
          className="flex-1 bg-transparent border-none outline-none resize-none py-3 px-1 text-[15.5px] text-text-main placeholder-text-disabled/50 max-h-[300px] font-medium"
          minRows={1}
          disabled={isSending || editingIndex !== null}
        />

        {/* Action Button: Send or Stop */}
        {isSending ? (
          <button 
            onClick={stopResponse}
            className="p-3 rounded-full mb-1 bg-red-900/20 text-red-400 hover:bg-red-900/40 transition-colors border border-red-900/30"
            type="button"
            title="Dừng phản hồi"
          >
            <Square size={20} fill="currentColor" />
          </button>
        ) : (
          <button 
            onClick={handleSend}
            disabled={!inputBuffer.trim() || isSending}
            className={`p-3 rounded-full mb-1 transition-all duration-300 ${
              inputBuffer.trim() && !isSending 
                ? "bg-gradient-to-br from-emerald-primary to-emerald-accent text-emerald-base shadow-[0_0_15px_rgba(0,255,180,0.5)] scale-110 hover:scale-115" 
                : "bg-emerald-base text-text-disabled opacity-50"
            }`}
            type="button"
          >
            <ArrowUp size={20} strokeWidth={3} />
          </button>
        )}
      </div>
    </div>
  );
}
