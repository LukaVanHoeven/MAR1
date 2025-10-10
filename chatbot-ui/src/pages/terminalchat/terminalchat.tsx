import Terminal, { ColorMode, TerminalOutput } from "react-terminal-ui";
import { useEffect } from "react";
import { WebSocketClient } from "@/lib/websocket";
import { useChatbotStore } from "@/lib/chatbotstore";

const TerminalChat = () => {
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
        flushBuffer();
      } else if (data === "[ENDCONVO]") {
        setShowReset(true);
      } else {
        appendToBuffer(data);
      }
    };

    const cleanup = WebSocketClient.onMessage(handleMessage);
    return cleanup;
  }, [flushBuffer, setShowReset, appendToBuffer]);

  const handleInput = (input: string) => {
    addMessage(input, "user");
    WebSocketClient.sendMessage(input);
  };
  return (
    <div className="w-full h-full overflow-hidden ">
      <Terminal colorMode={ColorMode.Dark} onInput={handleInput}>
        {messages.map((message) => (
          <TerminalOutput key={message.id}>
            {message.role === "user" ? `> ${message.content}` : message.content}
          </TerminalOutput>
        ))}
      </Terminal>
      {showReset && (
        <button
          className="absolute top-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          onClick={() => {
            clearMessages();
          }}
        >
          Reset
        </button>
      )}
    </div>
  );
};

export default TerminalChat;
