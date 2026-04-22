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
        chunks: [],
        maxScore: 0
      };
    }
    
    if (ref.article) {
      acc[key].articles.add(ref.article);
    }
    acc[key].chunks.push(ref);
    // Track max score for sorting groups
    const score = (ref as any).score || 0;
    if (score > acc[key].maxScore) acc[key].maxScore = score;
    
    return acc;
  }, {} as Record<string, any>))
    .sort((a: any, b: any) => (b.maxScore || 0) - (a.maxScore || 0));

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
                      {group.articles.size > 1 && chunk.article && (
                        <div className="font-black not-italic mb-1 text-emerald-primary/70 text-[11px] uppercase tracking-tighter">
                            {chunk.article}
                        </div>
                      )}
                      <div className="opacity-90 space-y-1.5 mt-2">
                        {(() => {
                          const textStr = chunk?.text_preview;
                          if (!textStr || typeof textStr !== 'string') return <span className="opacity-90">"{textStr}..."</span>;

                          const parts = textStr.split(/(Văn bản:|Lĩnh vực:|Điều khoản:|Nội dung:)/);
                          if (parts.length <= 1) return <span className="opacity-90">"{textStr}..."</span>;

                          return parts.reduce((acc: any[], part: string, idx: number, arr: string[]) => {
                            if (["Văn bản:", "Lĩnh vực:", "Điều khoản:", "Nội dung:"].includes(part)) {
                              let contentValue: React.ReactNode = arr[idx + 1] || "";
                              if (part === "Nội dung:" && typeof contentValue === 'string' && contentValue.includes(";")) {
                                const contentString = contentValue;
                                const contentParts = contentString.split(";").filter((s) => s.trim() !== "");
                                contentValue = (
                                  <ul className="list-disc leading-relaxed pl-4 space-y-1 mt-1">
                                    {contentParts.map((s, i) => (
                                      <li key={i}>{s.trim()}{i < contentParts.length - 1 ? ";" : ""}</li>
                                    ))}
                                  </ul>
                                );
                              }

                              acc.push(
                                <div key={`${cidx}-${idx}`} className="flex flex-col sm:flex-row gap-1 sm:gap-2 border-b border-emerald-primary/5 pb-2 pt-1">
                                  <span className="font-bold text-emerald-primary/90 whitespace-nowrap shrink-0">{part}</span>
                                  <div className="text-text-dim relative top-[1px] w-full">{contentValue}</div>
                                </div>
                              );
                            }
                            return acc;
                          }, []);
                        })()}
                      </div>
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
