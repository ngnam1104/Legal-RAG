"use client";
import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message, useChat } from "@/contexts/ChatContext";
import LegalReference from "./LegalReference";
import { Edit2, X, Check, Send } from "lucide-react";
import TextareaAutosize from "react-textarea-autosize";

export function MessageItem({ message, index, isLastUserMessage }: { 
  message: Message, 
  index: number,
  isLastUserMessage?: boolean
}) {
  const isUser = message.role === "user";
  const { setEditingIndex, editingIndex, sendMessage, isSending, cancelEdit } = useChat();
  const [editValue, setEditValue] = useState(message.content);
  const isEditing = editingIndex === index;

  const handleUpdate = () => {
    if (!editValue.trim() || editValue.trim() === message.content.trim() || isSending) return;
    sendMessage(editValue, index);
  };

  // Đồng bộ nội dung khi bắt đầu sửa
  useEffect(() => {
    if (isEditing) {
      setEditValue(message.content);
    }
  }, [isEditing, message.content]);

  return (
    <div className={`flex w-full mb-10 group transition-all duration-500 ${isUser ? "justify-end" : "justify-start"}`}>
      {/* Bot Avatar */}
      {!isUser && (
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-primary to-emerald-accent flex-shrink-0 flex items-center justify-center text-emerald-base shadow-[0_0_20px_rgba(0,255,180,0.4)] mr-4 mt-1 border border-white/20 transform rotate-3 hover:rotate-0 transition-transform">
          <span className="font-black text-xs tracking-tighter">VN</span>
        </div>
      )}

      <div className={`max-w-[88%] relative ${isUser ? "" : "glass-emerald p-6 rounded-2xl glow-border shadow-[0_10px_30px_rgba(0,0,0,0.3)]"}`}>
        {/* User Message Bubble */}
        {isUser ? (
          isEditing ? (
            <div className="flex flex-col w-full min-w-[300px] md:min-w-[500px] animate-in fade-in zoom-in-95 duration-300">
              <TextareaAutosize
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                autoFocus
                className="w-full bg-emerald-surface/50 text-text-main px-6 py-4 rounded-2xl border-2 border-emerald-primary/40 focus:border-emerald-accent outline-none shadow-2xl resize-none text-[15px] leading-relaxed transition-all"
              />
              <div className="flex justify-end gap-3 mt-3">
                <button 
                  onClick={() => {
                      cancelEdit();
                      setEditValue(message.content);
                  }}
                  className="px-4 py-2 text-xs font-bold uppercase tracking-widest text-text-disabled hover:text-white hover:bg-white/5 rounded-xl transition-all border border-white/5"
                >
                  Hủy
                </button>
                <button 
                  onClick={handleUpdate}
                  disabled={!editValue.trim() || editValue.trim() === message.content.trim() || isSending}
                  className="flex items-center gap-2 px-5 py-2 text-xs font-black uppercase tracking-widest bg-gradient-to-br from-emerald-primary to-emerald-accent text-emerald-base rounded-xl shadow-[0_5px_15px_rgba(0,255,180,0.3)] hover:scale-105 transition-all disabled:opacity-30 disabled:grayscale disabled:scale-100"
                >
                  <Send size={14} />
                  Cập nhật
                </button>
              </div>
            </div>
          ) : (
            <div className="relative">
              <div className="bg-emerald-surface text-text-main px-6 py-4 rounded-3xl rounded-tr-md text-[15px] leading-relaxed break-words border border-emerald-primary/20 shadow-xl">
                {message.content}
              </div>
              {isLastUserMessage && !isSending && (
                <button 
                  onClick={() => setEditingIndex(index)}
                  className="absolute -left-12 top-1/2 -translate-y-1/2 p-2.5 text-text-disabled hover:text-emerald-accent hover:bg-emerald-primary/10 rounded-full opacity-0 group-hover:opacity-100 transition-all"
                  title="Sửa câu hỏi"
                >
                  <Edit2 size={18} />
                </button>
              )}
            </div>
          )
        ) : (
          /* Assistant Message Block */
          <div className="text-text-main text-[15.5px] leading-relaxed">
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
            
            {/* References Accordion */}
            {message.references && message.references.length > 0 && (
              <div className="mt-8 pt-8 border-t border-emerald-primary/10">
                <LegalReference refs={message.references} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatArea() {
  const { messages, isLoading, isSending, editingIndex, activeMode } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  if (isLoading) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-text-dim">
        <div className="w-12 h-12 border-4 border-emerald-primary/20 border-t-emerald-accent rounded-full animate-spin shadow-[0_0_20px_rgba(0,255,180,0.2)] mb-6"></div>
        <p className="font-black uppercase tracking-[0.4em] text-[10px] glow-text animate-pulse">Truy phục dữ liệu...</p>
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-4 py-20 animate-in fade-in zoom-in duration-700">
        <div className="w-28 h-28 bg-gradient-to-br from-emerald-primary to-emerald-accent rounded-[2.5rem] flex items-center justify-center text-emerald-base text-6xl mb-10 shadow-[0_15px_50px_rgba(0,255,180,0.4)] border-2 border-white/20 transform hover:scale-110 transition-transform cursor-default">
          ⚖️
        </div>
        <h1 className="text-5xl font-black text-emerald-accent mb-6 tracking-tighter glow-text uppercase italic">
          Chatbot Pháp Luật
        </h1>
        <p className="text-text-dim max-w-xl mx-auto text-xl font-medium leading-relaxed opacity-90">
          Phân tích, tra cứu và giải đáp mọi vấn đề pháp lý với sức mạnh AI chuyên nghiệp.
        </p>
        
        {activeMode !== 'GENERAL_CHAT' && (
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-2xl px-4">
              {["Luật Đất Đai 2024", "Luật Nhà Ở", "Nghị định 15/2020", "Bộ luật Lao động"].map(tag => (
                  <div key={tag} className="px-4 py-3 rounded-2xl bg-emerald-surface/30 border border-emerald-primary/10 text-xs font-bold text-emerald-primary/80 hover:border-emerald-accent/40 hover:text-emerald-accent transition-all cursor-default">
                      # {tag}
                  </div>
              ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {messages.map((msg, idx) => {
        const lastUserIdx = messages.length - 1 - [...messages].reverse().findIndex(m => m.role === 'user');
        const isActuallyLastUser = msg.role === 'user' && idx === lastUserIdx;

        return (
          <MessageItem 
            key={idx} 
            index={idx}
            message={msg} 
            isLastUserMessage={isActuallyLastUser} 
          />
        );
      })}

      {/* Modern Skeletal Loading during send */}
      {isSending && (
         <div className="flex justify-start mb-10 animate-in fade-in slide-in-from-left-4 duration-500">
            <div className="w-10 h-10 rounded-xl bg-emerald-primary/10 flex-shrink-0 flex items-center justify-center text-emerald-accent mr-4 mt-1 border border-emerald-primary/10">
                <div className="w-5 h-5 border-2 border-emerald-accent/20 border-t-emerald-accent rounded-full animate-spin"></div>
            </div>
            <div className="glass-emerald p-6 rounded-2xl border border-emerald-primary/20 flex flex-col gap-3 min-w-[200px]">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-emerald-accent rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-emerald-accent rounded-full animate-bounce [animation-delay:0.2s]"></div>
                    <div className="w-2 h-2 bg-emerald-accent rounded-full animate-bounce [animation-delay:0.4s]"></div>
                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-accent/60 ml-2">Đang xử lý...</span>
                </div>
                <div className="space-y-2">
                    <div className="h-2 w-full bg-emerald-primary/5 rounded-full animate-pulse"></div>
                    <div className="h-2 w-5/6 bg-emerald-primary/5 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                    <div className="h-2 w-4/6 bg-emerald-primary/5 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                </div>
            </div>
         </div>
      )}
      
      <div ref={bottomRef} className="h-10" />
    </div>
  );
}
