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
        <label className="flex items-center gap-2">
          <span className="uppercase tracking-tighter">AI Focus:</span>
          <select 
            value={settings.provider}
            onChange={(e) => setSettings({ provider: e.target.value })}
            className="bg-emerald-surface border border-emerald-primary/30 rounded-lg px-2 py-1 outline-none focus:border-emerald-accent text-text-main transition-all"
          >
            <option value="groq">Groq (Llama 3)</option>
            <option value="ollama">Ollama (Local)</option>
            <option value="gemini">Gemini (Pro)</option>
          </select>
        </label>

        <label className="flex items-center gap-2 cursor-pointer group">
          <div className="relative flex items-center">
            <input 
              type="checkbox" 
              checked={settings.use_reflection}
              onChange={(e) => setSettings({ use_reflection: e.target.checked })}
              className="rounded bg-emerald-surface border-emerald-primary/30 text-emerald-primary focus:ring-emerald-accent w-4 h-4 transition-all"
            />
          </div>
          <span title="Tự kiểm tra chống ảo giác" className="group-hover:text-emerald-accent transition-colors">Reflection</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer group">
          <div className="relative flex items-center">
            <input 
              type="checkbox" 
              checked={settings.use_rerank}
              onChange={(e) => setSettings({ use_rerank: e.target.checked })}
              className="rounded bg-emerald-surface border-emerald-primary/30 text-emerald-primary focus:ring-emerald-accent w-4 h-4 transition-all"
            />
          </div>
          <span title="Xếp hạng bằng Cross-Encoder" className="group-hover:text-emerald-accent transition-colors">Rerank</span>
        </label>
      </div>
    </header>
  );
}
