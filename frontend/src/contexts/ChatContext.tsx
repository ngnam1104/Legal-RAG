"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useRouter, usePathname } from 'next/navigation';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  attached_file?: { id: string; name: string };
  references?: Array<{
    title: string;
    article: string;
    score: number;
    text_preview: string;
    chunk_id?: string;
    document_number?: string;
    url?: string;
  }>;
}

export interface Session {
  id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
}

export interface ChatSettings {
  llm_preset: string; // 'groq_8b', 'groq_70b', 'gemini', 'ollama'
  top_k: number;
  use_reflection: boolean;
  use_rerank: boolean;
}

interface ChatContextProps {
  // State
  currentSessionId: string | null;
  sessions: Session[];
  messages: Message[];
  settings: ChatSettings;
  inputBuffer: string;
  isPendingEdit: boolean;
  editingIndex: number | null;
  isLoading: boolean;
  isSending: boolean;
  isIngesting: boolean;
  stagedFile: { id: string, name: string } | null;
  processingSteps: { text: string; time: string }[];
  
  // Actions
  setInputBuffer: (text: string) => void;
  setCurrentSessionId: (id: string | null) => void;
  setSettings: (settings: Partial<ChatSettings>) => void;
  createNewSession: () => void;
  deleteSession: (id: string) => Promise<void>;
  renameSession: (id: string, newTitle: string) => Promise<void>;
  sendMessage: (query: string, editIndex?: number) => Promise<void>;
  stopResponse: () => void;
  setEditingIndex: (index: number | null) => void;
  cancelEdit: () => void;
  uploadFile: (file: File) => Promise<any>;
  ingestFile: () => Promise<any>;
  clearStagedFile: () => void;
  syncConflict: (docNumbers: string[], fileId?: string, filename?: string) => Promise<any>;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const ChatContext = createContext<ChatContextProps | undefined>(undefined);

export const ChatProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputBuffer, setInputBuffer] = useState("");
  const [lastFileId, setLastFileId] = useState<string | null>(null);
  const [isPendingEdit, setIsPendingEdit] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [stagedFile, setStagedFile] = useState<{ id: string, name: string } | null>(null);
  const [processingSteps, setProcessingSteps] = useState<{ text: string; time: string }[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isCreatingSessionRef = useRef(false);
  const [settings, setSettingsState] = useState<ChatSettings>({
    llm_preset: 'groq_70b',
    top_k: 3,
    use_reflection: true,
    use_rerank: false, // Mặc định tắt Reranker để ưu tiên tốc độ xử lý Local
  });

  const setSettings = (newSettings: Partial<ChatSettings>) => {
    setSettingsState(prev => ({ ...prev, ...newSettings }));
  };

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/sessions`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  }, []);

  const fetchMessages = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      } else {
        setMessages([]);
      }
    } catch (e) {
      console.error("Failed to fetch messages", e);
      setMessages([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const hasInitializedRef = useRef(false);

  // Load sessions on init
  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // AUTO-LOAD LATEST SESSION on first page load - DISABLED per user request
  // We want to ALWAYS start a New Chat on refresh.
  useEffect(() => {
    if (!hasInitializedRef.current && sessions.length > 0) {
      // setCurrentSessionId(sessions[0].id); // DISABLED
      hasInitializedRef.current = true;
    }
  }, [sessions]);

  const router = useRouter();
  const pathname = usePathname();

  // Load messages when session changes
  useEffect(() => {
    if (currentSessionId) {
      if (isCreatingSessionRef.current) {
        isCreatingSessionRef.current = false;
        return; // Skip fetching if we just created this session locally
      }
      fetchMessages(currentSessionId);
    } else {
      setMessages([]);
    }
  }, [currentSessionId, fetchMessages]);

  const createNewSession = () => {
    // 1. Dừng ngay LLM nếu đang phản hồi
    stopResponse();

    // 2. Force a fresh state locally
    setMessages([]);
    setInputBuffer("");
    setStagedFile(null);
    setLastFileId(null);
    isCreatingSessionRef.current = false;
    
    // 3. Navigate home
    router.push("/");
    
    // 4. Refresh sidebar to ensure any previous session is up to date
    fetchSessions();
  };

  const renameSession = async (id: string, newTitle: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${id}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        fetchSessions();
      }
    } catch (e) {
      console.error("Failed to rename session", e);
    }
  };

  const deleteSession = async (id: string) => {
    try {
      await fetch(`${API_BASE_URL}/sessions/${id}`, { method: 'DELETE' });
      if (currentSessionId === id) {
        router.push("/");
      }
      fetchSessions();
    } catch (e) {
      console.error("Failed to delete session", e);
    }
  };

  const sendMessage = async (query: string, editIndex?: number) => {
    setIsSending(true);
    
    try {
      let sid = currentSessionId;
      if (!sid) {
        sid = crypto.randomUUID();
        isCreatingSessionRef.current = true;
        setCurrentSessionId(sid);
        if (pathname === "/") {
          router.push(`/${sid}`);
        }
        // Refresh sessions to show the new empty/loading session in sidebar if needed
        fetchSessions();
      }

      const payload = {
        session_id: sid,
        query,
        mode: 'AUTO',
        file_path: lastFileId ? lastFileId : null,
        llm_preset: settings.llm_preset,
        top_k: settings.top_k,
        use_reflection: settings.use_reflection,
        use_rerank: settings.use_rerank,
      };

      // --- STATE CẬP NHẬT (Xử lý Edit + Append Query) ---
      setMessages(prev => {
        let baseMessages = prev;
        
        // 1. Nếu là Edit: Cắt bỏ các tin nhắn từ vị trí edit trở đi
        if (isPendingEdit || editIndex !== undefined) {
          const targetIdx = editIndex !== undefined ? editIndex : prev.length - 1;
          baseMessages = prev.slice(0, targetIdx);
        }

        // 2. Dọn dẹp tin nhắn "Đã dừng"
        const filtered = baseMessages.filter(msg => msg.content !== "_Đã dừng phản hồi._");
        
        // 3. Append tin nhắn mới
        return [...filtered, { role: 'user', content: query, attached_file: stagedFile ? { ...stagedFile } : undefined }];
      });

      // Reset edit states ngay lập tức
      setEditingIndex(null);
      setIsPendingEdit(false);

      // Gọi API xóa trên Server nếu là Edit (Silent)
      if ((isPendingEdit || editIndex !== undefined) && sid) {
        fetch(`${API_BASE_URL}/sessions/${sid}/last-turn`, { method: 'DELETE' }).catch(e => console.error("Sync edit error:", e));
      }

      // Clear staged file locally
      clearStagedFile();

      // Setup abort controller
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setProcessingSteps([{ text: "🚀 Bắt đầu...", time: new Date().toLocaleTimeString() }]);

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi API (${res.status}): ${errText}` }]);
        return;
      }

      // --- STREAMING HANDLER (SSE) ---
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) throw new Error("No reader available");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === "step") {
                setProcessingSteps(prev => [...prev, { text: data.content, time: new Date().toLocaleTimeString() }]);
              } 
              else if (data.type === "final") {
                const final = data.content;
                setMessages(prev => [...prev, {
                  role: 'assistant',
                  content: final.answer,
                  references: final.references || []
                }]);
                
                if (final.title) {
                  fetchSessions();
                }
              }
              else if (data.type === "error") {
                setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi Hệ thống: ${data.content}` }]);
              }
              else if (data.type === "cancelled") {
                 setMessages(prev => [...prev, { role: 'assistant', content: "_Đã dừng phản hồi._" }]);
              }
            } catch (e) {
              // Ignore partial JSON or malformed SSE lines
            }
          }
        }
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        setMessages(prev => [...prev, { role: 'assistant', content: "_Đã dừng phản hồi._" }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi kết nối: ${e.message}` }]);
      }
    } finally {
      setIsSending(false);
      abortControllerRef.current = null;
    }
  };

  const stopResponse = () => {
    console.log("🛑 Client: stopResponse triggered");
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      // Chúng ta sẽ setIsSending(false) ngay lập tức để UI cập nhật trạng thái đã dừng
      setIsSending(false);
    }
  };

  const cancelEdit = () => {
    setIsPendingEdit(false);
    setEditingIndex(null);
    setInputBuffer("");
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    if (currentSessionId) {
      formData.append("session_id", currentSessionId);
    }

    try {
      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      
      if (data.file_id && data.filename) {
        setStagedFile({ id: data.file_id, name: data.filename });
        const ext = data.filename.split('.').pop();
        setLastFileId(`${data.file_id}.${ext}`);
      }
      return data;
    } catch (e) {
      console.error("Upload error", e);
      throw e;
    }
  };

  const ingestFile = async () => {
    if (!stagedFile || !currentSessionId) return;
    
    setIsIngesting(true);
    try {
      const res = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          file_id: stagedFile.id,
          filename: stagedFile.name
        }),
      });
      
      const data = await res.json();
      if (res.ok) {
        if (data.status === "success" || data.status === "duplicate") {
          setIsIngesting(false);
          return data;
        }
      }
      throw new Error(data.detail || "Ingestion failed");
    } catch (e) {
      setIsIngesting(false);
      throw e;
    }
  };

  const clearStagedFile = () => {
    setStagedFile(null);
    setLastFileId(null);
  };

  const syncConflict = async (docNumbers: string[], fileId?: string, filename?: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/documents/sync-conflict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_numbers_to_disable: docNumbers,
          new_file_id: fileId,
          new_filename: filename
        }),
      });
      return await res.json();
    } catch (e) {
      console.error("Sync conflict error", e);
      throw e;
    }
  };

  const contextValue = useMemo(() => ({
    currentSessionId,
    sessions,
    messages,
    settings,
    inputBuffer,
    isPendingEdit,
    editingIndex,
    isLoading,
    isSending,
    isIngesting,
    stagedFile,
    processingSteps,
    setInputBuffer,
    setCurrentSessionId,
    setSettings,
    createNewSession,
    deleteSession,
    renameSession,
    sendMessage,
    stopResponse,
    setEditingIndex,
    cancelEdit,
    uploadFile,
    ingestFile,
    clearStagedFile,
    syncConflict
  }), [
    currentSessionId, sessions, messages, settings,
    inputBuffer, isPendingEdit, editingIndex, isLoading,
    isSending, isIngesting, stagedFile, processingSteps, fetchSessions, fetchMessages,
    createNewSession, deleteSession, renameSession, sendMessage,
    stopResponse, uploadFile, ingestFile, clearStagedFile, syncConflict
  ]);

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) throw new Error("useChat must be used within ChatProvider");
  return context;
};
