"use client";
import React, { useState } from "react";
import { ChevronDown, ChevronRight, Scale, ExternalLink, FileText, Database } from "lucide-react";
import { Message } from "@/contexts/ChatContext";

export default function LegalReference({ refs }: { refs: NonNullable<Message['references']> }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!refs || refs.length === 0) return null;

  // Group by document_number (fallback to title)
  const groupedRefs = Object.values(refs.reduce((acc, ref) => {
    const docNum = ref.document_number || "";
    const title = ref.title || 'Tài liệu không tên';
    const key = docNum ? `${docNum}-${title}` : title;
    
    if (!acc[key]) {
      acc[key] = {
        title: title,
        document_number: docNum,
        url: ref.url,
        isUpload: ref.document_number === "File Upload" || (!ref.document_number && !ref.url),
        chunks: [],
        maxScore: 0
      };
    }
    
    // Deduplicate chunks within the same group by text content
    const isDuplicate = acc[key].chunks.some((c: any) => 
      (c.text_preview === ref.text_preview) || (ref.chunk_id && c.chunk_id === ref.chunk_id)
    );

    if (!isDuplicate) {
      acc[key].chunks.push(ref);
      const score = (ref as any).score || 0;
      if (score > acc[key].maxScore) acc[key].maxScore = score;
    }
    
    return acc;
  }, {} as Record<string, any>))
    .sort((a: any, b: any) => (b.maxScore || 0) - (a.maxScore || 0));

  return (
    <div className="mt-8 border-t-2 border-emerald-primary/30 pt-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 mb-4 w-full text-left group"
      >
        <div className="h-8 w-1 bg-emerald-accent rounded-full shadow-[0_0_10px_#10b981]" />
        <h3 className="text-sm font-black uppercase tracking-[0.25em] text-emerald-accent glow-text flex-1">
          Cơ sở pháp lý chi tiết
        </h3>
        {isOpen ? (
          <ChevronDown size={18} className="text-emerald-primary" />
        ) : (
          <ChevronRight size={18} className="text-emerald-primary opacity-50 group-hover:opacity-100" />
        )}
      </button>

      {isOpen && (
        <div className="grid grid-cols-1 gap-4">
          {groupedRefs.map((group, idx) => (
            <div key={idx} className="glass-emerald rounded-2xl border border-emerald-primary/20 overflow-hidden shadow-lg hover:border-emerald-primary/50 transition-all group/doc">
              {/* Header: Document Title */}
              <div className="p-4 bg-emerald-primary/5 border-b border-emerald-primary/10 flex items-start gap-3">
                <div className="mt-1 p-2 rounded-lg bg-emerald-primary/10 group-hover/doc:bg-emerald-primary/20 transition-colors">
                  {group.isUpload ? (
                    <FileText size={18} className="text-amber-400" />
                  ) : (
                    <Scale size={18} className="text-emerald-primary shadow-glow" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[9px] font-black uppercase tracking-widest px-2 py-0.5 rounded bg-white/5 text-text-disabled">
                      {group.isUpload ? "Tài liệu tải lên" : "Cơ sở dữ liệu hệ thống"}
                    </span>
                    {group.document_number && group.document_number !== "File Upload" && (
                      <span className="text-[9px] font-black uppercase tracking-widest px-2 py-0.5 rounded bg-emerald-primary/20 text-emerald-accent">
                        {group.document_number}
                      </span>
                    )}
                  </div>
                  <h4 className="text-[13px] font-bold text-text-main leading-tight">
                    {group.url ? (
                      <a href={group.url} target="_blank" rel="noopener noreferrer" className="hover:text-emerald-accent transition-colors flex items-center gap-1.5">
                        {group.title}
                        <ExternalLink size={12} className="opacity-50" />
                      </a>
                    ) : (
                      group.title
                    )}
                  </h4>
                </div>
              </div>

              {/* Content: Chunks/Articles */}
              <div className="p-4 space-y-4 bg-emerald-base/20">
                {group.chunks.map((chunk: any, cidx: number) => (
                  <div key={cidx} className="relative pl-4 border-l-2 border-emerald-primary/20">
                    {chunk.article && (
                      <div className="text-[11px] font-black text-emerald-accent mb-2 flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-accent" />
                        {chunk.article}
                      </div>
                    )}
                    <div className="text-[13px] leading-relaxed text-text-dim font-medium italic bg-black/10 p-3 rounded-xl border border-white/5 whitespace-pre-wrap">
                      {chunk.text_preview ? (
                        chunk.text_preview.length > 1000 ? chunk.text_preview.substring(0, 1000) + "..." : chunk.text_preview
                      ) : (
                        <span className="opacity-50 italic">Không có nội dung chi tiết.</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
