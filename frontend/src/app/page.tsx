import Sidebar from "@/components/layout/Sidebar";
import SettingsHeader from "@/components/layout/SettingsHeader";
import ChatArea from "@/components/chat/ChatArea";
import ChatInput from "@/components/chat/ChatInput";
import ModeSelector from "@/components/chat/ModeSelector";

export default function Home() {
  return (
    <div className="flex h-screen w-full bg-emerald-base text-text-main font-sans">
      {/* LEFT SIDEBAR: Lịch sử Chat */}
      <Sidebar />

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col h-full bg-emerald-base/50 md:rounded-l-3xl shadow-[-10px_0_30px_-5px_rgba(0,255,180,0.1)] border-l border-emerald-primary/10 overflow-hidden relative">
         <SettingsHeader />
         
        {/* Mode Selector - Sticky at the top */}
        <div className="py-4 bg-emerald-base/40 backdrop-blur-md border-b border-emerald-primary/10 z-10 sticky top-0">
          <ModeSelector />
        </div>

        {/* Chat Area Scrollable */}
        <div className="flex-1 overflow-y-auto w-full max-w-5xl mx-auto px-4 py-6 pb-32">
           <ChatArea />
        </div>

        {/* Floating Input Area */}
        <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-emerald-base via-emerald-base/90 to-transparent pt-10 pb-6 pointer-events-none">
          <div className="max-w-3xl mx-auto px-4 pointer-events-auto">
            <ChatInput />
          </div>
        </div>
      </main>
    </div>
  );
}
