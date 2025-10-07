// websocket.ts
class WebSocketSingleton {
  private static instance: WebSocketSingleton;
  public socket: WebSocket;

  private constructor() {
    this.socket = new WebSocket("wss://ws.lukahoef.nl");

    this.socket.onopen = () => {
      console.log("WebSocket connection established");
    };

    this.socket.onclose = () => {
      console.log("WebSocket connection closed");
    };

    this.socket.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }

  public static getInstance(): WebSocketSingleton {
    if (!WebSocketSingleton.instance) {
      WebSocketSingleton.instance = new WebSocketSingleton();
    }
    return WebSocketSingleton.instance;
  }

  public sendMessage(msg: string) {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(msg);
    } else {
      console.warn("WebSocket not open yet");
    }
  }

  public onMessage(callback: (data: string) => void) {
    this.socket.onmessage = (event) => callback(event.data);
  }
}

export const WebSocketClient = WebSocketSingleton.getInstance();
