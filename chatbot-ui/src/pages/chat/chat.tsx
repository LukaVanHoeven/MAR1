import { ChatInput } from "@/components/custom/chatinput";
import {
  PreviewMessage,
  ThinkingMessage,
} from "../../components/custom/message";
import { useScrollToBottom } from "@/components/custom/use-scroll-to-bottom";
import { useState, useEffect } from "react";
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import { WebSocketClient } from "@/lib/websocket";
import { ChatMessage, useChatbotStore } from "@/lib/chatbotstore";

export function Chat() {
  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const {
    messages,
    showReset,
    addMessage,
    appendToBuffer,
    flushBuffer,
    setShowReset,
    clearMessages,
  } = useChatbotStore();

  useEffect(() => {
    const handleMessage = (data: string) => {
      if (data === "[END]") {
        setIsLoading(false);
        flushBuffer();
      } else if (data === "[ENDCONVO]") {
        setIsLoading(false);
        setShowReset(true);
      } else {
        appendToBuffer(data);
      }
    };

    const cleanup = WebSocketClient.onMessage(handleMessage);
    return cleanup;
  }, [flushBuffer, setShowReset, appendToBuffer]);

  const resetMessages = () => {
    clearMessages();
    setShowReset(false);
  };

  const handleSubmit = () => {
    if (!question.trim()) return;

    addMessage(question, "user");
    WebSocketClient.sendMessage(question);

    setQuestion("");
    setIsLoading(true);
  };

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header />
      <div
        className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4"
        ref={messagesContainerRef}
      >
        {messages.length === 0 && <Overview />}
        {messages.map((message: ChatMessage) => (
          <PreviewMessage key={message.id} message={message} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div
          ref={messagesEndRef}
          className="shrink-0 min-w-[24px] min-h-[24px]"
        />
      </div>
      <button
        className={`mx-auto mb-2 text-sm text-gray-500 hover:underline ${
          showReset ? "block" : "hidden"
        }`}
        onClick={resetMessages}
      >
        Start New Conversation
      </button>
      <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
