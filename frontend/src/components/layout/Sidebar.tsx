"use client";

import { useState, useRef } from "react";
import { MessageSquarePlus, MessageSquare, Trash2, Edit2, Check, X } from "lucide-react";
import { useChat } from "@/contexts/ChatContext";
import { useRouter } from "next/navigation";

export default function Sidebar() {
  const router = useRouter();
  const { sessions, currentSessionId, setCurrentSessionId, createNewSession, deleteSession, renameSession } = useChat();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const handleStartEdit = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditTitle(currentTitle);
  };

  const handleSaveEdit = async (id: string) => {
    if (editTitle.trim()) {
      await renameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <aside className="w-64 bg-emerald-base h-full flex flex-col border-r border-emerald-primary/10">
      <div className="p-4">
        <button 
          onClick={createNewSession}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-br from-emerald-primary to-emerald-accent hover:brightness-110 text-emerald-base py-2.5 px-4 rounded-xl font-bold transition-all shadow-[0_0_15px_rgba(0,200,140,0.3)] hover:shadow-[0_0_25px_rgba(0,255,180,0.5)]"
        >
          <MessageSquarePlus size={18} />
          <span>Phiên chat mới</span>
        </button>
      </div>

      <div className="px-4 pb-2 text-xs font-bold text-text-dim uppercase tracking-widest opacity-80">
        Lịch sử Chat
      </div>

      <div className="flex-1 overflow-y-auto px-2 space-y-1 custom-scrollbar">
        {sessions.length === 0 ? (
          <p className="text-sm text-text-disabled px-2 py-4 italic">Chưa có phiên chat nào.</p>
        ) : (
          sessions.map((sess) => {
            const isActive = sess.id === currentSessionId;
            const isEditing = editingId === sess.id;

            return (
              <div 
                key={sess.id}
                className={`group flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer transition-all border ${
                  isActive 
                    ? "bg-emerald-surface/80 text-emerald-accent font-semibold border-emerald-primary/40 shadow-[inset_0_0_10px_rgba(0,255,180,0.1)]" 
                    : "text-text-dim hover:bg-emerald-surface/40 hover:text-text-main border-transparent"
                }`}
                onClick={() => {
                  if (!isEditing) {
                    router.push(`/${sess.id}`);
                  }
                }}
              >
                <div className="flex items-center gap-3 overflow-hidden flex-1">
                  <MessageSquare size={16} className={isActive ? "text-emerald-accent glow-text" : "text-text-disabled"} />
                  {isEditing ? (
                    <input 
                      autoFocus
                      className="text-sm bg-emerald-base border border-emerald-primary/50 rounded px-1 py-0.5 outline-none w-full text-text-main glow-border"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit(sess.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      onBlur={() => handleSaveEdit(sess.id)}
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span className="text-sm truncate w-full">
                      {sess.title || "Phiên chat mới"}
                    </span>
                  )}
                </div>
                
                <div className="flex items-center gap-1">
                  {!isEditing && (
                    <>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartEdit(sess.id, sess.title);
                        }}
                        className={`p-1 rounded-md text-text-disabled hover:text-emerald-accent hover:bg-emerald-primary/10 transition-colors ${
                          isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                        }`}
                        title="Đổi tên"
                      >
                        <Edit2 size={14} />
                      </button>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSession(sess.id);
                        }}
                        className={`p-1 rounded-md text-text-disabled hover:text-red-400 hover:bg-red-900/20 transition-colors ${
                          isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                        }`}
                        title="Xóa phiên"
                      >
                        <Trash2 size={14} />
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      <div className="p-4 border-t border-emerald-primary/10 text-[10px] text-text-disabled uppercase tracking-tighter text-center font-medium">
        Vibecoding Chatbot
      </div>
    </aside>
  );
}
