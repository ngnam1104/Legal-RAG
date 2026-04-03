"use client";
import React, { useState } from "react";
import { ChevronDown, ChevronRight, Scale, ExternalLink } from "lucide-react";
import { Message } from "@/contexts/ChatContext";

export default function LegalReference({ refs }: { refs: NonNullable<Message['references']> }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!refs || refs.length === 0) return null;

  // Group by document_number (fallback to title)
  const groupedRefs = Object.values(refs.reduce((acc, ref) => {
    const key = ref.document_number || ref.title || 'unknown';
    
    if (!acc[key]) {
      acc[key] = {
        title: ref.title || 'Tài liệu không tên',
        document_number: ref.document_number,
        url: ref.url,
        articles: new Set<string>(),
        chunks: []
      };
    }
    
    if (ref.article) {
      acc[key].articles.add(ref.article);
    }
    acc[key].chunks.push(ref);
    
    return acc;
  }, {} as Record<string, any>));

  return (
    <div className="mt-4 border border-emerald-primary/10 rounded-2xl overflow-hidden glass-emerald shadow-[0_4px_30px_rgba(0,0,0,0.2)]">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 w-full p-4 text-sm font-bold text-emerald-accent hover:bg-emerald-primary/10 transition-all group"
      >
        <Scale size={18} className="text-emerald-primary glow-text group-hover:scale-110 transition-transform" />
        <span className="uppercase tracking-widest text-[11px]">Căn cứ pháp lý ({groupedRefs.length} văn bản)</span>
        {isOpen ? <ChevronDown size={18} className="ml-auto" /> : <ChevronRight size={18} className="ml-auto" />}
      </button>
      
      {isOpen && (
        <div className="p-4 border-t border-emerald-primary/10 text-sm space-y-4 bg-emerald-base/30">
          {groupedRefs.map((group, idx) => {
            const articlesArray = Array.prototype.slice.call(Array.from(group.articles));
            const articlesText = articlesArray.length > 0 ? articlesArray.join(', ') : 'Nội dung chung';
            
            return (
              <div key={idx} className="bg-emerald-surface/50 p-4 rounded-xl border border-emerald-primary/20 shadow-inner group/card hover:border-emerald-primary/40 transition-all">
                <div className="font-bold text-text-main mb-3 flex items-start gap-2">
                  <span className="text-emerald-primary font-black mt-0.5">{idx + 1}.</span>
                  {group.url ? (
                    <a href={group.url} target="_blank" rel="noopener noreferrer" className="text-emerald-accent hover:text-emerald-glow hover:underline inline-flex items-start gap-1 transition-colors glow-text">
                      <span className="leading-tight">{group.title}</span>
                      <ExternalLink size={14} className="inline flex-shrink-0 mt-0.5" />
                    </a>
                  ) : (
                    <span className="leading-tight">{group.title}</span>
                  )}
                </div>
                
                <div className="flex flex-wrap items-center gap-2 text-[10px] text-text-dim mb-4 mb-3">
                  <span className="bg-emerald-primary/20 text-emerald-accent px-3 py-1 rounded-full font-black tracking-wider uppercase border border-emerald-primary/10">
                    {articlesText}
                  </span>
                </div>
                
                <div className="space-y-3">
                  {group.chunks.map((chunk: any, cidx: number) => (
                    <div key={cidx} className="text-text-dim/80 leading-relaxed italic border-l-2 border-emerald-primary/30 pl-4 break-words text-[13px] relative group/chunk hover:border-emerald-accent transition-colors">
                      {chunk.article && (
                        <div className="font-black not-italic mb-1 text-emerald-primary/70 text-[11px] uppercase tracking-tighter">
                            {chunk.article}
                        </div>
                      )}
                      <span className="opacity-90">"{chunk.text_preview}..."</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
