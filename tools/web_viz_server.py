import asyncio
import json
import base64
import os
import time
from typing import List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks

# Import SessionManager
import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
from tools.session_manager import SessionManager

app = FastAPI()

# serve frontend static files from /static
app.mount("/static", StaticFiles(directory="web"), name="static")
# serve output files (videos, images)
Path("out").mkdir(exist_ok=True)
app.mount("/out", StaticFiles(directory="out"), name="out")

# in-memory queue of events
event_queue: asyncio.Queue = asyncio.Queue()

# Session Manager Instance
# Hardcoded ffmpeg path for now based on user context
FFMPEG_PATH = r"C:\Users\HP\Downloads\ffmpeg\bin\ffmpeg.exe"
session_mgr = SessionManager(ffmpeg_path=FFMPEG_PATH)

# Processing state for background jobs
_proc_jobs = {}

try:
    import openai
    _have_openai = True
except Exception:
    _have_openai = False
# runtime stats
stats = {
    'received': 0,
    'batches_broadcast': 0,
    'last_batch_size': 0,
}


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


async def stats_logger():
    while True:
        try:
            print(f"[web_viz_server] stats received={stats['received']} batches_broadcast={stats['batches_broadcast']} last_batch_size={stats['last_batch_size']}")
        except Exception:
            pass
        await asyncio.sleep(2.0)


# Global loop reference
main_loop = None


@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_running_loop()

    # Configure session manager callback
    def callback_wrapper(data, img_bytes):
        if main_loop:
            if img_bytes:
                data['image_base64'] = base64.b64encode(img_bytes).decode('utf-8')
            main_loop.call_soon_threadsafe(event_queue.put_nowait, data)

    session_mgr.set_callback(callback_wrapper)

    # start background tasks
    asyncio.create_task(broadcaster_task())
    asyncio.create_task(stats_logger())


@app.post("/api/session/start")
async def start_session():
    try:
        sid = session_mgr.start(output_dir_base="out")
        return {"ok": True, "session_id": sid}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/session/stop")
async def stop_session():
    try:
        path = session_mgr.stop()
        return {"ok": True, "path": path}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def _read_jsonl(path):
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _summarize_text(text: str, max_points: int = 6):
    """Attempt to summarize text into knowledge points using OpenAI if available,
    otherwise fall back to simple keyword extraction."""
    if not text or text.strip() == "":
        return []
    if _have_openai and os.getenv('OPENAI_API_KEY'):
        try:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            prompt = (
                "Extract up to {} concise knowledge points from the following classroom transcript. "
                "Return a JSON array of short strings, each a single knowledge point.\n\n".format(max_points)
            ) + text
            resp = openai.ChatCompletion.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            content = resp['choices'][0]['message']['content']
            # Try to parse JSON array from response
            try:
                pts = json.loads(content)
                if isinstance(pts, list):
                    return [str(p).strip() for p in pts if p]
            except Exception:
                # Fallback: split by lines and return top lines
                lines = [l.strip('-* \t') for l in content.splitlines() if l.strip()]
                return lines[:max_points]
        except Exception:
            pass

    # Fallback simple extractor: top frequent non-stopwords
    stopwords = set(["the","is","and","to","a","of","in","that","it","for","on","with","as","are","this","be","by","an","or","from","at","we","you","they","he","she"]) 
    words = [w.strip('.,:;?()[]"').lower() for w in text.split() if len(w) > 2]
    freq = {}
    for w in words:
        if w in stopwords: continue
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])[:max_points]
    return [w for w, c in items]


@app.post('/api/session/process')
async def process_session(request: Request, background_tasks: BackgroundTasks):
    """Process the last session outputs: produce transcript, stats and knowledge points.
    Returns immediate job id while work runs in background.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = body.get('session_id') or session_mgr.session_id
    if not sid:
        return JSONResponse({"ok": False, "error": "no session_id available"}, status_code=400)

    session_dir = Path('out') / sid
    if not session_dir.exists():
        return JSONResponse({"ok": False, "error": f"session dir not found: {session_dir}"}, status_code=404)

    job_id = f"job_{int(time.time())}"
    _proc_jobs[job_id] = {"status": "queued", "session_id": sid}

    def _do_work(job_id, session_dir):
        try:
            _proc_jobs[job_id]['status'] = 'running'
            # Find video file (prefer mp4 then avi)
            video_candidates = list(session_dir.glob('session.mp4')) + list(session_dir.glob('*.mp4')) + list(session_dir.glob('temp_video.avi'))
            video_path = video_candidates[0] if video_candidates else None

            # Build transcript from asr.jsonl if present
            asr_path = session_dir / 'asr.jsonl'
            transcript_txt = session_dir / 'transcript.txt'
            asr_segments = []
            if asr_path.exists():
                asr_segments = _read_jsonl(asr_path)
                # concatenate
                with open(transcript_txt, 'w', encoding='utf-8') as fh:
                    for seg in asr_segments:
                        text = seg.get('text') if isinstance(seg, dict) else ''
                        if not text and isinstance(seg, str):
                            text = seg
                        fh.write((text or '').strip() + '\n')
            else:
                # try to find any *.asr.jsonl or asr*.jsonl
                found = list(session_dir.glob('*.asr.jsonl')) + list(session_dir.glob('*.asr'))
                if found:
                    asr_segments = _read_jsonl(found[0])
                    with open(transcript_txt, 'w', encoding='utf-8') as fh:
                        for seg in asr_segments:
                            text = seg.get('text') if isinstance(seg, dict) else ''
                            fh.write((text or '').strip() + '\n')

            # Parse cv events to build per-student non-awake intervals
            events_path = session_dir / 'cv_events.jsonl'
            faces_path = session_dir / 'faces.jsonl'
            events = _read_jsonl(events_path) if events_path.exists() else []
            faces = _read_jsonl(faces_path) if faces_path.exists() else []

            per_student = {}
            # Build intervals from events (DROWSY_START / DROWSY_END, LOOKING_DOWN_START/END)
            for ev in events:
                sid_ev = str(ev.get('student_id', ev.get('track_id', 'unknown')))
                typ = ev.get('type')
                ts = float(ev.get('ts', 0.0))
                if sid_ev not in per_student:
                    per_student[sid_ev] = {'intervals': []}
                if typ and typ.endswith('_START'):
                    # store start; we will look for matching END
                    per_student[sid_ev].setdefault('open', []).append({'type': typ.replace('_START',''), 'start': ts, 'end': None})
                elif typ and typ.endswith('_END'):
                    name = typ.replace('_END','')
                    # close the most recent open interval of same type
                    opens = per_student[sid_ev].get('open', [])
                    for o in reversed(opens):
                        if o['type'] == name and o['end'] is None:
                            o['end'] = ts
                            break

            # finalize intervals
            for sid_ev, info in per_student.items():
                intervals = []
                for o in info.get('open', []):
                    start = o.get('start', 0.0)
                    end = o.get('end') if o.get('end') is not None else (max([ev.get('ts',0) for ev in events]) if events else start+1.0)
                    if end <= start:
                        end = start + 0.5
                    intervals.append({'type': o.get('type'), 'start': float(start), 'end': float(end)})
                per_student[sid_ev]['intervals'] = intervals
                per_student[sid_ev].pop('open', None)

            # Associate ASR segments to intervals
            for sid_ev, info in per_student.items():
                for it in info['intervals']:
                    # gather ASR segments that overlap interval
                    txts = []
                    for seg in asr_segments:
                        s = float(seg.get('start', 0.0))
                        e = float(seg.get('end', s + 2.0))
                        # overlap if seg midpoint inside interval or any overlap
                        mid = (s+e)/2.0
                        if (mid >= it['start'] and mid <= it['end']) or (s < it['end'] and e > it['start']):
                            txts.append(seg.get('text',''))
                    joined = '\n'.join([t for t in txts if t])
                    it['asr_text'] = joined
                    # summarize into knowledge points
                    it['knowledge_points'] = _summarize_text(joined or '')

            # Save stats file
            stats_out = session_dir / 'stats.json'
            with open(stats_out, 'w', encoding='utf-8') as fh:
                json.dump({'session_id': session_dir.name, 'video': str(video_path.name) if video_path else None, 'transcript': str(transcript_txt.name) if transcript_txt.exists() else None, 'per_student': per_student}, fh, ensure_ascii=False, indent=2)

            # mark complete
            _proc_jobs[job_id]['status'] = 'done'
            _proc_jobs[job_id]['result'] = {
                'video': f"/out/{session_dir.name}/{video_path.name}" if video_path else None,
                'transcript': f"/out/{session_dir.name}/{transcript_txt.name}" if transcript_txt.exists() else None,
                'stats': f"/out/{session_dir.name}/{stats_out.name}",
            }
        except Exception as exc:
            _proc_jobs[job_id]['status'] = 'error'
            _proc_jobs[job_id]['error'] = str(exc)

    # enqueue background task
    background_tasks.add_task(_do_work, job_id, session_dir)

    return JSONResponse({"ok": True, "job_id": job_id})


@app.get('/api/session/process/status')
async def process_status(job_id: str):
    info = _proc_jobs.get(job_id)
    if not info:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return JSONResponse({"ok": True, "job": info})


@app.get("/api/session/status")
async def session_status():
    return {
        "is_recording": session_mgr.is_recording,
        "session_id": session_mgr.session_id
    }


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
