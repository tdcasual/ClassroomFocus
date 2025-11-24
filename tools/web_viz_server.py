import asyncio
import json
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# serve frontend static files from /static
app.mount("/static", StaticFiles(directory="web"), name="static")

# in-memory queue of events
event_queue: asyncio.Queue = asyncio.Queue()

# simple runtime stats
stats = {
    'received': 0,
    'batches_broadcast': 0,
    'last_batch_size': 0,
}


async def stats_logger():
    """Periodically log server stats for debugging/observability."""
    while True:
        try:
            print(f"[web_viz_server] stats received={stats['received']} batches_broadcast={stats['batches_broadcast']} last_batch_size={stats['last_batch_size']}")
        except Exception:
            pass
        await asyncio.sleep(2.0)

# set of active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self.active.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, message: str):
        coros = []
        for ws in list(self.active):
            coros.append(ws.send_text(message))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

manager = ConnectionManager()


async def broadcaster_task():
    """Background task that batches events and broadcasts every 100ms."""
    BATCH_INTERVAL = 0.1
    while True:
        batch = []
        try:
            # wait for first item
            item = await asyncio.wait_for(event_queue.get(), timeout=BATCH_INTERVAL)
            batch.append(item)
        except asyncio.TimeoutError:
            # nothing arrived in interval
            pass

        # drain queue without blocking
        while True:
            try:
                item = event_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        if batch:
            stats['last_batch_size'] = len(batch)
            stats['batches_broadcast'] += 1
            payload = json.dumps({"type": "batch", "events": batch})
            await manager.broadcast(payload)

        await asyncio.sleep(BATCH_INTERVAL)


@app.on_event("startup")
async def startup():
    # start background broadcaster
    asyncio.create_task(broadcaster_task())
    # start stats logger
    asyncio.create_task(stats_logger())


@app.post("/push")
async def push(request: Request):
    """POST JSON event or a list of events to push to connected clients.
    Example body: {"type":"asr_segment", ...}  or [{...}, {...}]
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    events = data if isinstance(data, list) else [data]
    for e in events:
        # ensure timestamp
        if isinstance(e, dict) and "ts" not in e:
            e["ts"] = asyncio.get_event_loop().time()
        await event_queue.put(e)
        stats['received'] += 1

    return JSONResponse({"ok": True, "queued": len(events)})


@app.get('/stats')
async def get_stats():
    return JSONResponse(stats)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # keep connection alive; client messages ignored for now
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.web_viz_server:app", host="0.0.0.0", port=8000, reload=False)
