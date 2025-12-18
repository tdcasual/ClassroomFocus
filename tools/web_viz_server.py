import asyncio
import json
import base64
import os
import time
import re
from typing import List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, RedirectResponse
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
FFMPEG_PATH = os.getenv("FFMPEG_PATH") or "ffmpeg"
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


@app.get("/")
async def root():
    return RedirectResponse(url="/static/viz.html")


@app.get("/api/sessions")
async def list_sessions():
    base = Path("out")
    sessions = []
    try:
        for p in base.iterdir():
            if not p.is_dir():
                continue
            sid = p.name
            st = p.stat()
            stats_path = p / "stats.json"
            transcript_path = p / "transcript.txt"
            video_path = p / "session.mp4"
            sessions.append({
                "session_id": sid,
                "mtime": float(getattr(st, "st_mtime", 0.0)),
                "has_stats": stats_path.exists(),
                "has_transcript": transcript_path.exists(),
                "has_video": video_path.exists() or any(p.glob("*.mp4")),
            })
    except Exception:
        sessions = []
    sessions.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
    return {"ok": True, "sessions": sessions}


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

    # Fallback extractor:
    # 1) If CJK text dominates, extract frequent 2-8 char phrases.
    cjk = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    if len(cjk) >= 3:
        freq = {}
        for tok in cjk:
            freq[tok] = freq.get(tok, 0) + 1
        items = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
        points = []
        for tok, _ in items:
            if any(tok in p or p in tok for p in points):
                continue
            points.append(tok)
            if len(points) >= max_points:
                break
        return points

    # 2) Otherwise, extract frequent tokens (space-delimited languages)
    stopwords = set([
        "the", "is", "and", "to", "a", "of", "in", "that", "it", "for", "on", "with", "as", "are", "this", "be", "by", "an", "or", "from", "at", "we", "you", "they", "he", "she",
    ])
    words = [w.strip('.,:;?()[]"').lower() for w in text.split() if len(w) > 2]
    freq = {}
    for w in words:
        if w in stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])[:max_points]
    return [w for w, _ in items]


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

            def _as_text(obj) -> str:
                if obj is None:
                    return ""
                if isinstance(obj, str):
                    return obj.strip()
                if isinstance(obj, dict):
                    if obj.get("text"):
                        return str(obj.get("text")).strip()
                    words = obj.get("words")
                    if isinstance(words, list):
                        return "".join(str(w.get("text", "")) for w in words).strip()
                return str(obj).strip()

            def _maybe_generate_asr_from_audio() -> None:
                """Optional: if `asr.jsonl` is missing but we have `temp_audio.wav`,
                try to run DashScope file transcription to produce ASR segments.
                """
                asr_path = session_dir / "asr.jsonl"
                if asr_path.exists():
                    return
                wav_in = session_dir / "temp_audio.wav"
                if not wav_in.exists():
                    return

                # Map DASH_SCOPE_API_KEY -> DASHSCOPE_API_KEY for compatibility
                if os.getenv("DASH_SCOPE_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
                    os.environ["DASHSCOPE_API_KEY"] = os.environ["DASH_SCOPE_API_KEY"]
                if not os.getenv("DASHSCOPE_API_KEY"):
                    return

                try:
                    from asr.asr_client import transcribe_file
                except Exception:
                    return

                wav_for_asr = wav_in
                try:
                    import soundfile as sf
                    import numpy as np
                    data, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)
                    if data.size == 0:
                        return
                    mono = data[:, 0] if data.ndim == 2 else data.reshape(-1)
                    mono = np.clip(mono, -1.0, 1.0)
                    pcm = (mono * 32767.0).astype(np.int16)
                    pcm_wav = session_dir / "asr_audio_pcm16.wav"
                    sf.write(str(pcm_wav), pcm, sr, subtype="PCM_16")
                    wav_for_asr = pcm_wav
                except Exception:
                    wav_for_asr = wav_in

                _proc_jobs[job_id]["status"] = "running:asr"
                events = []
                cur_sec = 0.0
                last_sec = -1

                def progress_hook(sec: float):
                    nonlocal cur_sec, last_sec
                    cur_sec = float(sec or 0.0)
                    sec_i = int(cur_sec)
                    if sec_i != last_sec:
                        last_sec = sec_i
                        _proc_jobs[job_id]["status"] = f"running:asr {sec_i}s"

                def time_base():
                    return cur_sec

                def on_sentence(ev: dict):
                    if isinstance(ev, dict):
                        events.append(ev)

                try:
                    transcribe_file(
                        api_key=None,
                        wav_path=str(wav_for_asr),
                        time_base=time_base,
                        on_sentence=on_sentence,
                        progress_hook=progress_hook,
                    )
                except Exception:
                    _proc_jobs[job_id]["status"] = "running"
                    return

                events.sort(key=lambda e: float(e.get("ts", 0.0)))
                segments = []
                for idx, ev in enumerate(events):
                    start = float(ev.get("ts", 0.0))
                    nxt = events[idx + 1] if idx + 1 < len(events) else None
                    end = float(nxt.get("ts", start + 2.0)) if nxt else (start + 2.0)
                    segments.append({
                        "start": float(start),
                        "end": float(max(end, start + 0.2)),
                        "text": _as_text(ev.get("text")),
                        "raw": ev.get("raw"),
                    })

                try:
                    with open(asr_path, "w", encoding="utf-8") as fh:
                        for seg in segments:
                            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
                except Exception:
                    return

            # Load/Generate ASR segments
            _maybe_generate_asr_from_audio()
            _proc_jobs[job_id]["status"] = "running:load_asr"
            transcript_txt = session_dir / "transcript.txt"
            asr_segments = []
            try:
                from analysis.teacher_labeler import load_asr_jsonl
            except Exception:
                load_asr_jsonl = None

            # Prefer session_dir/asr.jsonl; else fall back to any *.asr.jsonl
            asr_candidates = []
            if (session_dir / "asr.jsonl").exists():
                asr_candidates.append(session_dir / "asr.jsonl")
            asr_candidates.extend(session_dir.glob("*.asr.jsonl"))

            for p in asr_candidates:
                try:
                    if load_asr_jsonl:
                        asr_segments = load_asr_jsonl(str(p))
                    else:
                        # minimal compatibility: accept already-normalized segment dicts
                        asr_segments = _read_jsonl(p)
                    if asr_segments:
                        break
                except Exception:
                    continue

            if asr_segments:
                try:
                    with open(transcript_txt, "w", encoding="utf-8") as fh:
                        for seg in asr_segments:
                            if isinstance(seg, dict):
                                fh.write((_as_text(seg.get("text")) + "\n").strip() + "\n")
                            else:
                                fh.write((_as_text(seg) + "\n").strip() + "\n")
                except Exception:
                    pass

            # Parse cv events to build per-student non-awake intervals
            _proc_jobs[job_id]["status"] = "running:cv_intervals"
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
            _proc_jobs[job_id]["status"] = "running:align_asr"
            for sid_ev, info in per_student.items():
                for it in info['intervals']:
                    # gather ASR segments that overlap interval
                    txts = []
                    for seg in asr_segments:
                        if not isinstance(seg, dict):
                            continue
                        s = float(seg.get('start', seg.get('ts', 0.0)))
                        e = float(seg.get('end', s + float(seg.get('duration', 2.0))))
                        # overlap if seg midpoint inside interval or any overlap
                        mid = (s+e)/2.0
                        if (mid >= it['start'] and mid <= it['end']) or (s < it['end'] and e > it['start']):
                            txts.append(_as_text(seg.get('text', '')))
                    joined = '\n'.join([t for t in txts if t])
                    it['asr_text'] = joined
                    # summarize into knowledge points
                    it['knowledge_points'] = _summarize_text(joined or '')

            # Save stats file
            _proc_jobs[job_id]["status"] = "running:write_stats"
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
