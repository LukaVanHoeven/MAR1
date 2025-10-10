// src/contexts/WebSocketProvider.tsx
import { createContext, useContext, useEffect, ReactNode } from "react";
import { WebSocketClient } from "@/lib/websocket";
import { useChatbotStore } from "@/lib/chatbotstore";

const WebSocketContext = createContext<null>(null);

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
  const { addMessage } = useChatbotStore();

  useEffect(() => {
    console.log("WebSocketProvider: Connecting...");
    WebSocketClient.connect();

    // Listen for initial message from server
    const cleanup = WebSocketClient.onMessage((data) => {
      // Check if this is an initial greeting (before any user interaction)
      const { messages } = useChatbotStore.getState();
      if (messages.length === 0 && data !== "[END]" && data !== "[ENDCONVO]") {
        addMessage(data, "assistant");
      }
    });

    return () => {
      console.log("WebSocketProvider: Disconnecting...");
      cleanup();
      WebSocketClient.disconnect();
    };
  }, [addMessage]);

  return (
    <WebSocketContext.Provider value={null}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => useContext(WebSocketContext);
