"use client";

import React from "react";
import { MessageSquare, FileText, AlertTriangle, Coffee } from "lucide-react";
import { useChat } from "@/contexts/ChatContext";

const modes = [
  {
    id: "GENERAL_CHAT",
    label: "Trò chuyện tự do",
    icon: <Coffee size={16} />,
    description: "Hỏi đáp mọi chủ đề thông thường không liên quan đến pháp luật",
  },
  {
    id: "LEGAL_QA",
    label: "Hỏi đáp thông thường",
    icon: <MessageSquare size={16} />,
    description: "Giải đáp các tình huống pháp lý dựa trên CSDL",
  },
  {
    id: "SECTOR_SEARCH",
    label: "Tìm kiếm liên quan",
    icon: <FileText size={16} />,
    description: "Tổng hợp danh sách văn bản và tóm tắt theo ngành",
  },
  {
    id: "CONFLICT_ANALYZER",
    label: "Phát hiện xung đột",
    icon: <AlertTriangle size={16} />,
    description:
      "Rà soát điểm trái luật giữa văn bản nội bộ với quy định nhà nước",
  },
  {
    id: "AUTO",
    label: "Tự động thông minh",
    icon: <span className="text-sm font-bold">AI</span>,
    description: "AI tự động nhận diện ý định và chọn luồng xử lý phù hợp",
  },
] as const;

export default function ModeSelector() {
  const { activeMode, setActiveMode, isSending } = useChat();

  return (
    <div className="flex flex-col items-center mb-6 w-full max-w-6xl mx-auto px-4">
      <div className="flex p-1 bg-emerald-surface/60 backdrop-blur-md rounded-2xl border border-emerald-primary/20 w-full shadow-[0_4px_20px_rgba(0,255,180,0.1)]">
        {modes.map((mode) => {
          const isActive = activeMode === mode.id;
          return (
            <button
              key={mode.id}
              onClick={() => !isSending && setActiveMode(mode.id)}
              disabled={isSending}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl text-sm font-bold transition-all duration-300 ${
                isActive
                  ? "bg-emerald-primary text-emerald-base shadow-[0_0_20px_rgba(0,200,140,0.4)] scale-[1.02]"
                  : "text-text-dim hover:text-emerald-accent hover:bg-emerald-primary/10"
              } ${isSending ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <span className={isActive ? "text-emerald-base" : "text-emerald-primary/60 group-hover:text-emerald-accent transition-colors"}>
                {mode.icon}
              </span>
              <span className="whitespace-nowrap">{mode.label}</span>
            </button>
          );
        })}
      </div>

      {/* Subtle Hint */}
      <p className="mt-2 text-[10px] text-text-disabled font-bold uppercase tracking-[0.2em] glow-text opacity-70">
        {modes.find((m) => m.id === activeMode)?.description}
      </p>
    </div>
  );
}
