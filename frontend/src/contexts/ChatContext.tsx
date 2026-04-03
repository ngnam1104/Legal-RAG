"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
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
  provider: string;
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
  activeMode: 'LEGAL_QA' | 'SECTOR_SEARCH' | 'CONFLICT_ANALYZER' | 'GENERAL_CHAT' | 'AUTO';
  inputBuffer: string;
  isPendingEdit: boolean;
  isLoading: boolean;
  isSending: boolean;
  isIngesting: boolean;
  stagedFile: { id: string, name: string } | null;
  
  // Actions
  setInputBuffer: (text: string) => void;
  setActiveMode: (mode: 'LEGAL_QA' | 'SECTOR_SEARCH' | 'CONFLICT_ANALYZER' | 'GENERAL_CHAT' | 'AUTO') => void;
  setCurrentSessionId: (id: string | null) => void;
  setSettings: (settings: Partial<ChatSettings>) => void;
  createNewSession: () => void;
  deleteSession: (id: string) => Promise<void>;
  renameSession: (id: string, newTitle: string) => Promise<void>;
  sendMessage: (query: string) => Promise<void>;
  stopResponse: () => void;
  editLastMessage: () => void;
  cancelEdit: () => void;
  uploadFile: (file: File) => Promise<any>;
  ingestFile: () => Promise<any>;
  clearStagedFile: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const ChatContext = createContext<ChatContextProps | undefined>(undefined);

export const ChatProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputBuffer, setInputBuffer] = useState("");
  const [activeMode, setActiveMode] = useState<'LEGAL_QA' | 'SECTOR_SEARCH' | 'CONFLICT_ANALYZER' | 'GENERAL_CHAT' | 'AUTO'>('AUTO');
  const [lastFileId, setLastFileId] = useState<string | null>(null);
  const [isPendingEdit, setIsPendingEdit] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [stagedFile, setStagedFile] = useState<{ id: string, name: string } | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isCreatingSessionRef = useRef(false);
  const [settings, setSettingsState] = useState<ChatSettings>({
    provider: 'groq',
    top_k: 3,
    use_reflection: true,
    use_rerank: true,
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

  // AUTO-LOAD LATEST SESSION on first page load
  useEffect(() => {
    if (!hasInitializedRef.current && sessions.length > 0 && currentSessionId === null) {
      setCurrentSessionId(sessions[0].id);
      hasInitializedRef.current = true;
    }
  }, [sessions, currentSessionId]);

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
    // Refresh sidebar to finalize previous session in history
    fetchSessions();
    setCurrentSessionId(null); // Just set to null, let sendMessage generate ID when user types
    setMessages([]);
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
        setCurrentSessionId(null);
        setMessages([]);
      }
      fetchSessions();
    } catch (e) {
      console.error("Failed to delete session", e);
    }
  };

  const sendMessage = async (query: string) => {
    let sid = currentSessionId;
    if (!sid) {
      sid = crypto.randomUUID();
      isCreatingSessionRef.current = true;
      setCurrentSessionId(sid);
    }

    const payload = {
      session_id: sid,
      query,
      mode: activeMode,
      file_path: lastFileId ? `backend/tmp_uploads/${lastFileId}` : null,
      provider: settings.provider,
      top_k: settings.top_k,
      use_reflection: settings.use_reflection,
      use_rerank: settings.use_rerank,
    };

    // If we're resending an edited message, sync with backend AND local state now
    if (isPendingEdit && sid) {
      try {
        await fetch(`${API_BASE_URL}/sessions/${sid}/last-turn`, { method: 'DELETE' });
        
        // Find last user message index to slice local state
        const lastUserIdx = [...messages].reverse().findIndex(m => m.role === 'user');
        if (lastUserIdx !== -1) {
            const actualIdx = messages.length - 1 - lastUserIdx;
            setMessages(prev => prev.slice(0, actualIdx));
        }
        
        setIsPendingEdit(false);
      } catch (e) {
        console.error("Failed to delete last turn in BE", e);
      }
    }

    setMessages(prev => [...prev, { role: 'user', content: query }]);
    setIsSending(true);

    // Setup abort controller
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (res.ok) {
        const data = await res.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          references: data.references || []
        }]);

        // Cập nhật lại danh sách session để hiển thị tiêu đề mới ở Sidebar
        if (data.title) {
          fetchSessions();
        }
      } else {
        const errText = await res.text();
        setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi API (${res.status}): ${errText}` }]);
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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const editLastMessage = () => {
    if (messages.length === 0 || isSending) return;
    
    // Find last user message
    const lastUserIdx = [...messages].reverse().findIndex(m => m.role === 'user');
    if (lastUserIdx === -1) return;
    
    const actualIdx = messages.length - 1 - lastUserIdx;
    const lastUserMsg = messages[actualIdx];
    
    setInputBuffer(lastUserMsg.content);
    setIsPendingEdit(true);
  };

  const cancelEdit = () => {
    setIsPendingEdit(false);
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
        if (data.status === "success") {
          // Manual clear staged file on successful completion will be handled by UI polling if needed
          // or just return the task_id
          return data;
        } else if (data.status === "duplicate") {
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

  return (
    <ChatContext.Provider value={{
      currentSessionId,
      sessions,
      messages,
      settings,
      activeMode,
      inputBuffer,
      isPendingEdit,
      isLoading,
      isSending,
      isIngesting,
      stagedFile,
      setInputBuffer,
      setActiveMode,
      setCurrentSessionId,
      setSettings,
      createNewSession,
      deleteSession,
      renameSession,
      sendMessage,
      stopResponse,
      editLastMessage,
      cancelEdit,
      uploadFile,
      ingestFile,
      clearStagedFile
    }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) throw new Error("useChat must be used within ChatProvider");
  return context;
};
