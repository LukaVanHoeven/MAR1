import { ChatInput } from "@/components/custom/chatinput";
import {
  PreviewMessage,
  ThinkingMessage,
} from "../../components/custom/message";
import { useScrollToBottom } from "@/components/custom/use-scroll-to-bottom";
import { useState, useEffect, useRef } from "react";
import { message } from "../../interfaces/interfaces";
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import { v4 as uuidv4 } from "uuid";
import { WebSocketClient } from "@/lib/websocket";

export function Chat() {
  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showReset, setShowReset] = useState<boolean>(false);
  const currentBufferRef = useRef("");
  const currentTraceIdRef = useRef("");

  useEffect(() => {
    WebSocketClient.onMessage((data) => {
      if (data === "[END]") {
        setIsLoading(false);
        const newMessage = currentBufferRef.current;
        setMessages((prev) => [
          ...prev,
          {
            content: newMessage,
            role: "assistant",
            id: currentTraceIdRef.current,
          },
        ]);
        currentBufferRef.current = "";
        currentTraceIdRef.current = "";
      } else if (data === "[ENDCONVO]") {
        setIsLoading(false);
        setShowReset(true);
      } else {
        currentBufferRef.current += data;
      }
    });
  }, []);

  const resetMessages = () => {
    setMessages([
      {
        content:
          "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?",
        role: "assistant",
        id: currentTraceIdRef.current,
      },
    ]);
    setShowReset(false);
  };
  const handleSubmit = () => {
    if (!question.trim()) return;

    const traceId = uuidv4();
    setMessages((prev) => [
      ...prev,
      { content: question, role: "user", id: traceId },
    ]);
    currentTraceIdRef.current = traceId;
    currentBufferRef.current = "";

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
        {messages.map((message, index) => (
          <PreviewMessage key={index} message={message} />
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
