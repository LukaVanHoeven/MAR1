// src/stores/chatbotStore.ts
import { create } from "zustand";

export interface ChatMessage {
  // Add 'export' here
  content: string;
  role: "user" | "assistant";
  id: string;
}
interface ChatbotStore {
  messages: ChatMessage[];
  currentBuffer: string;
  currentTraceId: string;
  showReset: boolean;

  // Actions
  addMessage: (content: string, role: "user" | "assistant") => void; // Changed parameter name
  appendToBuffer: (data: string) => void;
  flushBuffer: () => void;
  setTraceId: (id: string) => void;
  setShowReset: (show: boolean) => void;
  clearMessages: () => void;
}

export const useChatbotStore = create<ChatbotStore>((set, get) => ({
  messages: [],
  currentBuffer: "",
  currentTraceId: "",
  showReset: false,

  addMessage: (content, role) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          content,
          role, // Changed from 'type' to 'role'
          id: `${role}-${Date.now()}-${Math.random()}`,
        },
      ],
    })),

  appendToBuffer: (data) =>
    set((state) => ({
      currentBuffer: state.currentBuffer + data,
    })),

  flushBuffer: () => {
    const { currentBuffer, currentTraceId } = get();
    if (currentBuffer) {
      set((state) => ({
        messages: [
          ...state.messages,
          {
            content: currentBuffer,
            role: "assistant", // Changed from 'type' to 'role'
            id: currentTraceId || `assistant-${Date.now()}`,
          },
        ],
        currentBuffer: "",
        currentTraceId: "",
      }));
    }
  },

  setTraceId: (id) => set({ currentTraceId: id }),

  setShowReset: (show) => set({ showReset: show }),

  clearMessages: () =>
    set({
      messages: [
        {
          content:
            "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?",
          role: "assistant",
          id: "user-0",
        },
      ],
      currentBuffer: "",
      currentTraceId: "",
      showReset: false,
    }),
}));
