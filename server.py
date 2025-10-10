import asyncio
import websockets
import os
from src.restaurant_recommendation import recommendation

async def echo(websocket):  # Removed 'path' parameter as it's no longer needed in newer websockets versions
    try:
        dms = recommendation()
        dms.set_websocket(websocket)
        
        async for message in websocket:
            result = dms.utter(message)
            if result == "end_conversation":
                await websocket.send("[ENDCONVO]")
                dms.reset()
                
            # Echo the message back
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

async def main():
    print("WebSocket server starting", flush=True)
    
    # Create the server with CORS headers
    async with websockets.serve(
        echo,
        "0.0.0.0",
        int(os.environ.get('PORT', 8090))
    ) as server:
        print("WebSocket server running on port 8090", flush=True)
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())