import "./App.css";
import { Chat } from "./pages/chat/chat";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import TerminalChat from "./pages/terminalchat/terminalchat";
import Home from "./pages/home/home";
import { WebSocketProvider } from "./components/custom/websocketprovider";

function App() {
  return (
    <ThemeProvider>
      <Router>
        <WebSocketProvider>
          <div className="w-full h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/graphicalchat" element={<Chat />} />
              <Route path="/terminalchat" element={<TerminalChat />} />
            </Routes>
          </div>
        </WebSocketProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
