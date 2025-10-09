// src/lib/websocket.ts
class WebSocketSingleton {
  private static instance: WebSocketSingleton;
  public socket: WebSocket | null = null;
  private messageCallbacks: Set<(data: string) => void> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  private constructor() {
    // Don't auto-connect in constructor
  }

  public connect() {
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      console.log("WebSocket already connected or connecting");
      return;
    }

    this.socket = new WebSocket("wss://ws.lukahoef.nl");

    this.socket.onopen = () => {
      console.log("WebSocket connection established");
      this.reconnectAttempts = 0;
    };

    this.socket.onclose = () => {
      console.log("WebSocket connection closed");
      // Auto-reconnect
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
        this.reconnectTimeout = setTimeout(
          () => this.connect(),
          1000 * this.reconnectAttempts
        );
      }
    };

    this.socket.onerror = (err) => {
      console.error("WebSocket error:", err);
    };

    this.socket.onmessage = (event) => {
      this.messageCallbacks.forEach((callback) => callback(event.data));
    };
  }

  public disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  public static getInstance(): WebSocketSingleton {
    if (!WebSocketSingleton.instance) {
      WebSocketSingleton.instance = new WebSocketSingleton();
    }
    return WebSocketSingleton.instance;
  }

  public sendMessage(msg: string) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(msg);
    } else {
      console.warn(
        "WebSocket not open yet, current state:",
        this.socket?.readyState
      );
    }
  }

  public onMessage(callback: (data: string) => void) {
    this.messageCallbacks.add(callback);
    return () => {
      this.messageCallbacks.delete(callback);
    };
  }

  public removeAllListeners() {
    this.messageCallbacks.clear();
  }
}

export const WebSocketClient = WebSocketSingleton.getInstance();
