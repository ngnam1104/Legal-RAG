import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ChatProvider } from "@/contexts/ChatContext";
import { Toaster } from "react-hot-toast";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Vibecoding Chatbot Pháp Luật VN",
  description: "Tra cứu, phân tích và phát hiện xung đột văn bản pháp luật",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ChatProvider>
          {children}
          <Toaster position="top-right" />
        </ChatProvider>
      </body>
    </html>
  );
}
