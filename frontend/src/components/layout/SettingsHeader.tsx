"use client";

import { Settings } from "lucide-react";
import { useChat } from "@/contexts/ChatContext";

export default function SettingsHeader() {
  const { settings, setSettings } = useChat();

  return (
    <header className="h-14 border-b border-emerald-primary/10 flex items-center justify-between px-6 bg-emerald-base/60 backdrop-blur-md z-10">
      <div className="flex items-center gap-2 text-emerald-accent font-bold glow-text">
        <span className="text-xl">⚖️</span>
        <span className="tracking-widest uppercase text-sm">Chatbot Pháp Luật</span>
      </div>

      <div className="flex items-center gap-6 text-xs font-medium text-text-dim">
        <label className="flex items-center gap-2 cursor-pointer group">
          <span className="uppercase tracking-tighter group-hover:text-emerald-accent transition-colors">CĐ Reranker:</span>
          <div className="relative inline-flex items-center cursor-pointer">
            <input 
              type="checkbox" 
              className="sr-only peer" 
              checked={settings.use_rerank}
              onChange={(e) => setSettings({ use_rerank: e.target.checked })}
            />
            <div className="w-11 h-6 bg-emerald-surface border border-emerald-primary/30 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-text-dim after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-primary peer-checked:after:bg-emerald-base shadow-inner"></div>
            <span className="ml-2 text-[10px] font-bold uppercase text-emerald-accent/80">
              {settings.use_rerank ? "Chính xác cao" : "Tốc độ nhanh"}
            </span>
          </div>
        </label>

        <label className="flex items-center gap-2">
          <span className="uppercase tracking-tighter">AI Model:</span>
          <select 
            value="internal"
            onChange={() => {}}
            className="bg-emerald-surface border border-emerald-primary/30 rounded-lg px-2 py-1 outline-none focus:border-emerald-accent text-text-main transition-all pointer-events-none opacity-80"
          >
            <option value="internal">Llama 3.2 8B API</option>
          </select>
        </label>
      </div>
    </header>
  );
}
