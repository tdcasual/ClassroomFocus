import asyncio
import json
import base64
import os
import time
import re
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks
from dotenv import load_dotenv

# Import SessionManager
import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
from asr.dashscope_offline import DashScopeOfflineConfig, transcribe_wav_to_segments as dashscope_transcribe_wav_to_segments
from asr.xfyun_raasr import XfyunRaasrConfig, transcribe_wav as xfyun_raasr_transcribe_wav
from tools.session_manager import SessionManager
from tools.openai_compat import OpenAICompat
from tools.language_utils import normalize_language, llm_language_hint, asr_language_param, resolve_asr_language
from tools.model_config_utils import merge_redacted_model_cfg, merge_session_meta
from tools.device_probe import detect_device_profile, DeviceProfile, save_runtime_profile, list_profiles
from tools.thumb_utils import effective_refresh_sec
from analysis.inattentive_intervals import infer_not_visible_intervals, merge_inattentive_intervals
from config import load_web_config

import logging
logger = logging.getLogger(__name__)

app = FastAPI()

# Load local `.env` (keys/config), if present.
load_dotenv(PROJ_ROOT / ".env")

# Avoid confusing stale frontend assets during local iteration.
@app.middleware("http")
async def _no_cache_static_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path.startswith("/static/") and any(path.endswith(ext) for ext in (".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response

# Resolve important dirs relative to repo root so the server can be started from anywhere.
WEB_DIR = PROJ_ROOT / "web"
OUT_DIR = PROJ_ROOT / "out"

# serve frontend static files from /static
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
# serve output files (videos, images)
OUT_DIR.mkdir(exist_ok=True)
app.mount("/out", StaticFiles(directory=str(OUT_DIR)), name="out")

# Session Manager Instance
FFMPEG_PATH = os.getenv("FFMPEG_PATH") or "ffmpeg"
session_mgr = SessionManager(ffmpeg_path=FFMPEG_PATH)

# Processing state for background jobs
_proc_jobs = {}
# runtime stats
stats = {
    'received': 0,
    'batches_broadcast': 0,
    'last_batch_size': 0,
    'dropped_events': 0,
    'coalesced_updates': 0,
}

# ---- Device Profile Detection (at startup) ----
DEVICE_PROFILE: DeviceProfile = detect_device_profile()
logger.info(f"Device profile detected: {DEVICE_PROFILE.name} (constrained={DEVICE_PROFILE.is_constrained}, "
            f"rpi={DEVICE_PROFILE.is_rpi}, arch={DEVICE_PROFILE.cpu_arch}, mem={DEVICE_PROFILE.mem_gb:.1f}GB)")
# Save profile for other processes
save_runtime_profile(DEVICE_PROFILE, str(PROJ_ROOT))

WEB_CFG = load_web_config(DEVICE_PROFILE)
UI_MODE = WEB_CFG.ui_mode
LITE_MODE = UI_MODE == "lite"
LITE_STATUS_INTERVAL_SEC = WEB_CFG.lite.status_interval_sec
LITE_AUTO_START = WEB_CFG.lite.auto_start
LITE_AUTO_REPORT = WEB_CFG.lite.auto_report
LITE_DISABLE_VIDEO_WRITE = WEB_CFG.lite.disable_video_write
LITE_DISABLE_AUDIO = WEB_CFG.lite.disable_audio
LITE_WRITE_FACES = WEB_CFG.lite.write_faces
LITE_FACES_SAMPLE_SEC = WEB_CFG.lite.faces_sample_sec
LITE_AUTO_STOP_SEC = WEB_CFG.lite.auto_stop_sec
LITE_AUTO_UPLOAD = WEB_CFG.lite.auto_upload
LITE_ADAPTIVE_SCHEDULER = WEB_CFG.lite.adaptive_scheduler
LITE_RECOVERY_BACKOFF_SEC = WEB_CFG.lite.recovery_backoff_sec
LITE_RECOVERY_MAX_FAILS = WEB_CFG.lite.recovery_max_fails
LITE_HEARTBEAT_SEC = WEB_CFG.lite.heartbeat_sec
LITE_THUMBNAILS = WEB_CFG.lite.thumbnails
LITE_THUMB_INTERVAL_SEC = WEB_CFG.lite.thumb_interval_sec
LITE_THUMB_SIZE = WEB_CFG.lite.thumb_size
LITE_THUMB_QUALITY = WEB_CFG.lite.thumb_quality
LITE_THUMB_PAD = WEB_CFG.lite.thumb_pad
LITE_THUMB_REFRESH_SEC = WEB_CFG.lite.thumb_refresh_sec
LITE_THUMB_REFRESH_FRAMES = WEB_CFG.lite.thumb_refresh_frames
LITE_MAX_FACES = WEB_CFG.lite.max_faces
LITE_PROCESS_EVERY_N = WEB_CFG.lite.process_every_n
LITE_TARGET_FPS = WEB_CFG.lite.target_fps
LITE_INPUT_SCALE = WEB_CFG.lite.input_scale
LITE_DISABLE_CAMERA_SIMILARITY = WEB_CFG.lite.disable_camera_similarity
LITE_ADAPTIVE_CPU = WEB_CFG.lite.adaptive_cpu
LITE_ADAPTIVE_LATENCY_MS = WEB_CFG.lite.adaptive_latency_ms
LITE_ADAPTIVE_MAX_SKIP = WEB_CFG.lite.adaptive_max_skip

WEB_IDLE_THROTTLE_SEC = WEB_CFG.idle_throttle_sec
WEB_IDLE_PREVIEW_INTERVAL_SEC = WEB_CFG.idle_preview_interval_sec
WEB_IDLE_STATS_INTERVAL_SEC = WEB_CFG.idle_stats_interval_sec

if LITE_MODE:
    session_mgr.set_preview(
        enabled=LITE_THUMBNAILS,
        interval_sec=LITE_THUMB_INTERVAL_SEC if LITE_THUMBNAILS else None,
    )
    session_mgr.set_preview_thumbs(
        enabled=LITE_THUMBNAILS,
        size=LITE_THUMB_SIZE,
        quality=LITE_THUMB_QUALITY,
        pad=LITE_THUMB_PAD,
        refresh_sec=LITE_THUMB_REFRESH_SEC,
        refresh_frames=LITE_THUMB_REFRESH_FRAMES,
    )
    if LITE_DISABLE_VIDEO_WRITE:
        session_mgr.set_video_write(enabled=False)
    session_mgr.set_audio_enabled(not LITE_DISABLE_AUDIO)
    if LITE_WRITE_FACES and LITE_FACES_SAMPLE_SEC > 0:
        session_mgr.set_faces_write(enabled=True, sample_sec=LITE_FACES_SAMPLE_SEC)
    else:
        session_mgr.set_faces_write(enabled=False)
    lite_overrides = {
        "max_faces": int(LITE_MAX_FACES),
        "process_every_n": int(LITE_PROCESS_EVERY_N),
        "target_fps": float(LITE_TARGET_FPS),
        "input_scale": float(LITE_INPUT_SCALE),
        "compensate_camera_similarity": not bool(LITE_DISABLE_CAMERA_SIMILARITY),
    }
    session_mgr.set_device_profile(DEVICE_PROFILE, overrides=lite_overrides)
    if LITE_ADAPTIVE_SCHEDULER:
        session_mgr.set_adaptive_scheduler(
            enabled=True,
            target_cpu=float(LITE_ADAPTIVE_CPU),
            target_latency_ms=float(LITE_ADAPTIVE_LATENCY_MS),
            max_skip=int(LITE_ADAPTIVE_MAX_SKIP),
        )
    logger.info("UI mode: lite (preview disabled)")
else:
    logger.info("UI mode: full")

# in-memory queue of events
EVENT_QUEUE_MAXSIZE = int(getattr(WEB_CFG, "event_queue_maxsize", 2000) or 0)
event_queue: asyncio.Queue = (
    asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE) if EVENT_QUEUE_MAXSIZE > 0 else asyncio.Queue()
)

LAST_CLIENT_TS = time.time()
PREVIEW_DEFAULT_ENABLED = bool(session_mgr.preview_enabled)
PREVIEW_DEFAULT_INTERVAL = float(session_mgr.preview_interval_sec)
PREVIEW_DEFAULT_THUMBS = bool(session_mgr.preview_thumbs_enabled)
PREVIEW_DEFAULT_THUMB_CFG = {
    "size": int(session_mgr.preview_thumb_size),
    "quality": int(session_mgr.preview_thumb_quality),
    "pad": float(session_mgr.preview_thumb_pad),
    "refresh_sec": float(session_mgr.preview_thumb_refresh_sec),
    "refresh_frames": int(session_mgr.preview_thumb_refresh_frames),
}
PREVIEW_THROTTLED = False

# ---- Model config (in-memory defaults for next session) ----
_MODEL_CFG: Dict[str, Any] = {}

# ---- Lite UI state (for delta snapshots) ----
LITE_STATE = {
    "last_emit_ts": -1e9,
    "last_snapshot": {},
    "pending": False,
    "latest_faces": [],
    "latest_snapshot": {},
    "latest_ts": 0.0,
    "session_id": None,
    "last_diff": {"changed": {}, "removed": []},
    "thumbs": {},
    "thumb_updates": {},
    "thumb_ts": {},
}

COALESCE_TYPES = {"frame_data"}
COALESCE_CACHE: Dict[str, Dict[str, Any]] = {}
COALESCE_PENDING: Dict[str, bool] = {}
LITE_AUTOSTOP_TASK = None


def _reset_lite_state(session_id=None):
    LITE_STATE["last_emit_ts"] = -1e9
    LITE_STATE["last_snapshot"] = {}
    LITE_STATE["pending"] = False
    LITE_STATE["latest_faces"] = []
    LITE_STATE["latest_snapshot"] = {}
    LITE_STATE["latest_ts"] = 0.0
    LITE_STATE["session_id"] = session_id
    LITE_STATE["last_diff"] = {"changed": {}, "removed": []}
    LITE_STATE["thumbs"] = {}
    LITE_STATE["thumb_updates"] = {}
    LITE_STATE["thumb_ts"] = {}


def _mark_client_activity() -> None:
    global LAST_CLIENT_TS
    LAST_CLIENT_TS = time.time()


def _snapshot_from_faces(faces):
    snap = {}
    for f in faces:
        if not isinstance(f, dict):
            continue
        sid = f.get("track_id")
        if sid is None:
            sid = f.get("student_id")
        if sid is None:
            continue
        state = str(f.get("state") or "unknown")
        snap[str(sid)] = state
    return snap


def _diff_snapshots(prev, cur):
    changed = {}
    for sid, st in cur.items():
        if prev.get(sid) != st:
            changed[sid] = st
    removed = [sid for sid in prev.keys() if sid not in cur]
    return changed, removed


def _track_sort_key(item):
    sid = item.get("track_id")
    try:
        return (0, int(sid))
    except Exception:
        return (1, str(sid))


def _build_lite_faces(faces, ts):
    out = []
    for f in faces:
        if not isinstance(f, dict):
            continue
        sid = f.get("track_id")
        if sid is None:
            sid = f.get("student_id")
        if sid is None:
            continue
        out.append({
            "track_id": str(sid),
            "state": str(f.get("state") or "unknown"),
            "last_seen": float(f.get("ts") or ts or 0.0),
        })
    out.sort(key=_track_sort_key)
    return out


def _thumbs_for_snapshot(snapshot: Dict[str, Any]) -> Dict[str, str]:
    if not LITE_STATE.get("thumbs"):
        return {}
    out: Dict[str, str] = {}
    for sid in snapshot.keys():
        thumb = LITE_STATE["thumbs"].get(str(sid))
        if thumb:
            out[str(sid)] = thumb
    return out


def _lite_thumb_refresh_sec() -> float:
    preview_interval = float(getattr(session_mgr, "preview_interval_sec", 0.0) or 0.0)
    return effective_refresh_sec(LITE_THUMB_REFRESH_SEC, preview_interval, LITE_THUMB_REFRESH_FRAMES)


def _apply_thumb_updates(updates: Dict[str, Any], ts: float) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not updates:
        return out
    now_ts = float(ts)
    for sid, b64 in updates.items():
        if not b64:
            continue
        key = str(sid)
        val = str(b64)
        LITE_STATE.setdefault("thumbs", {})[key] = val
        LITE_STATE.setdefault("thumb_ts", {})[key] = now_ts
        out[key] = val
    return out


def _maybe_capture_thumbs(
    faces: List[Dict[str, Any]],
    img_bytes: Optional[bytes],
    ts: Optional[float] = None,
) -> Dict[str, str]:
    if not LITE_THUMBNAILS or not img_bytes or not faces:
        return {}
    refresh_sec = _lite_thumb_refresh_sec()
    now_ts = float(ts) if ts is not None else time.time()
    candidates = []
    for f in faces:
        if not isinstance(f, dict):
            continue
        sid = f.get("track_id")
        if sid is None:
            sid = f.get("student_id")
        if sid is None:
            continue
        sid = str(sid)
        last_ts = LITE_STATE.get("thumb_ts", {}).get(sid)
        if sid not in LITE_STATE.get("thumbs", {}):
            candidates.append((sid, f))
            continue
        if refresh_sec > 0 and (last_ts is None or (now_ts - float(last_ts)) >= refresh_sec):
            candidates.append((sid, f))
    if not candidates:
        return {}
    try:
        import numpy as np
        import cv2
    except Exception:
        return {}
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    h, w = img.shape[:2]
    updates: Dict[str, str] = {}
    for sid, f in candidates:
        bbox = f.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            continue
        try:
            nx, ny, nw, nh = [float(b) for b in bbox[:4]]
        except Exception:
            continue
        if nw <= 0 or nh <= 0:
            continue
        pad = float(LITE_THUMB_PAD)
        cx = nx + nw / 2.0
        cy = ny + nh / 2.0
        size = max(nw, nh) * (1.0 + pad * 2.0)
        sx0 = max(0.0, cx - size / 2.0)
        sy0 = max(0.0, cy - size / 2.0)
        sx1 = min(1.0, cx + size / 2.0)
        sy1 = min(1.0, cy + size / 2.0)
        x0 = int(max(0, min(w - 1, round(sx0 * w))))
        y0 = int(max(0, min(h - 1, round(sy0 * h))))
        x1 = int(max(1, min(w, round(sx1 * w))))
        y1 = int(max(1, min(h, round(sy1 * h))))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            thumb = cv2.resize(crop, (LITE_THUMB_SIZE, LITE_THUMB_SIZE), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), int(LITE_THUMB_QUALITY)])
        except Exception:
            continue
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode("utf-8")
        LITE_STATE.setdefault("thumbs", {})[sid] = b64
        LITE_STATE.setdefault("thumb_ts", {})[sid] = now_ts
        updates[sid] = b64
    return updates


def _lite_snapshot_payload():
    snapshot = LITE_STATE["latest_snapshot"] or LITE_STATE["last_snapshot"]
    faces = [{"track_id": sid, "state": st} for sid, st in snapshot.items()]
    faces.sort(key=_track_sort_key)
    payload = {"type": "lite_snapshot", "ts": float(time.time()), "faces": faces}
    thumbs = _thumbs_for_snapshot(snapshot)
    if thumbs:
        payload["thumbs"] = thumbs
    return payload


def _handle_lite_frame(
    data: Dict[str, Any], allow_emit: bool = True, img_bytes: Optional[bytes] = None
) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    try:
        ts = float(data.get("ts", 0.0))
    except Exception:
        ts = 0.0
    sid = session_mgr.session_id

    faces = data.get("faces") if isinstance(data.get("faces"), list) else []
    snapshot = _snapshot_from_faces(faces)
    data_thumbs = data.get("thumbs") if isinstance(data, dict) else None
    if isinstance(data_thumbs, dict) and data_thumbs:
        thumb_updates = _apply_thumb_updates(data_thumbs, ts)
    else:
        thumb_updates = _maybe_capture_thumbs(faces, img_bytes, ts=ts)
    if thumb_updates:
        LITE_STATE["thumb_updates"].update(thumb_updates)
        LITE_STATE["latest_faces"] = faces
        LITE_STATE["latest_snapshot"] = dict(snapshot)
        LITE_STATE["latest_ts"] = ts

    is_new_session = sid != LITE_STATE["session_id"]
    if not is_new_session and LITE_STATE["last_emit_ts"] > 0 and ts < (LITE_STATE["last_emit_ts"] - 1.0):
        is_new_session = True

    if is_new_session:
        _reset_lite_state(session_id=sid)
        LITE_STATE["latest_faces"] = faces
        LITE_STATE["latest_snapshot"] = dict(snapshot)
        LITE_STATE["latest_ts"] = ts
        LITE_STATE["last_snapshot"] = dict(snapshot)
        LITE_STATE["last_emit_ts"] = ts
        if allow_emit:
            payload = {
                "type": "lite_snapshot",
                "ts": ts,
                "faces": _build_lite_faces(faces, ts),
            }
            thumbs = _thumbs_for_snapshot(snapshot)
            if thumbs:
                payload["thumbs"] = thumbs
            if LITE_STATE["thumb_updates"]:
                LITE_STATE["thumb_updates"] = {}
            return payload
        return None

    if not allow_emit:
        LITE_STATE["latest_faces"] = faces
        LITE_STATE["latest_snapshot"] = dict(snapshot)
        LITE_STATE["latest_ts"] = ts
        LITE_STATE["last_snapshot"] = dict(snapshot)
        LITE_STATE["pending"] = False
        return None

    if snapshot != LITE_STATE["last_snapshot"]:
        LITE_STATE["pending"] = True
        LITE_STATE["latest_faces"] = faces
        LITE_STATE["latest_snapshot"] = snapshot
        LITE_STATE["latest_ts"] = ts
    else:
        LITE_STATE["pending"] = False

    if not LITE_STATE["pending"]:
        if LITE_STATE["thumb_updates"]:
            payload = {
                "type": "lite_delta",
                "ts": LITE_STATE["latest_ts"] or ts,
                "changed": {},
                "removed": [],
                "thumbs": dict(LITE_STATE["thumb_updates"]),
            }
            LITE_STATE["thumb_updates"] = {}
            LITE_STATE["last_emit_ts"] = LITE_STATE["latest_ts"] or ts
            LITE_STATE["last_diff"] = {"changed": {}, "removed": []}
            return payload
        return None

    if LITE_STATE["last_emit_ts"] < 0 or (ts - LITE_STATE["last_emit_ts"] >= LITE_STATUS_INTERVAL_SEC):
        changed, removed = _diff_snapshots(LITE_STATE["last_snapshot"], LITE_STATE["latest_snapshot"])
        if not changed and not removed and not LITE_STATE["thumb_updates"]:
            LITE_STATE["pending"] = False
            return None
        payload = {
            "type": "lite_delta",
            "ts": LITE_STATE["latest_ts"] or ts,
            "changed": changed,
            "removed": removed,
        }
        if LITE_STATE["thumb_updates"]:
            payload["thumbs"] = dict(LITE_STATE["thumb_updates"])
            LITE_STATE["thumb_updates"] = {}
        LITE_STATE["last_emit_ts"] = LITE_STATE["latest_ts"] or ts
        LITE_STATE["last_snapshot"] = dict(LITE_STATE["latest_snapshot"])
        LITE_STATE["pending"] = False
        LITE_STATE["last_diff"] = {"changed": dict(changed), "removed": list(removed)}
        return payload
    return None


def _is_coalesce_marker(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "_coalesce" and "key" in item


def _resolve_coalesced_item(item: Any) -> Optional[Dict[str, Any]]:
    if _is_coalesce_marker(item):
        key = item.get("key")
        if isinstance(key, str):
            COALESCE_PENDING[key] = False
            return COALESCE_CACHE.get(key)
        return None
    if isinstance(item, dict):
        return item
    return None


def _queue_put(payload: Dict[str, Any]) -> bool:
    try:
        event_queue.put_nowait(payload)
        return True
    except asyncio.QueueFull:
        try:
            dropped = event_queue.get_nowait()
            stats["dropped_events"] += 1
            if _is_coalesce_marker(dropped):
                key = dropped.get("key")
                if isinstance(key, str):
                    COALESCE_PENDING[key] = False
        except asyncio.QueueEmpty:
            pass
        try:
            event_queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            stats["dropped_events"] += 1
            return False


def _enqueue_event(payload: Dict[str, Any]) -> bool:
    if isinstance(payload, dict):
        ptype = payload.get("type")
        if isinstance(ptype, str) and ptype in COALESCE_TYPES:
            COALESCE_CACHE[ptype] = payload
            if not COALESCE_PENDING.get(ptype, False):
                marker = {"type": "_coalesce", "key": ptype}
                if _queue_put(marker):
                    COALESCE_PENDING[ptype] = True
                    stats["coalesced_updates"] += 1
                    return True
                return False
            stats["coalesced_updates"] += 1
            return True
    return _queue_put(payload)


def _queue_event_threadsafe(payload: Dict[str, Any]) -> None:
    if not main_loop:
        return
    try:
        main_loop.call_soon_threadsafe(_enqueue_event, payload)
    except Exception:
        pass


def _emit_error_event(message: str) -> None:
    if not message:
        return
    if not main_loop or not getattr(manager, "active", None):
        return
    payload = {"type": "error", "ts": time.time(), "error": str(message)}
    _queue_event_threadsafe(payload)


def _emit_lite_clear() -> None:
    if not main_loop or not getattr(manager, "active", None):
        return
    payload = {"type": "lite_snapshot", "ts": time.time(), "faces": []}
    _queue_event_threadsafe(payload)


async def _auto_stop_worker(session_id: str, delay_sec: float) -> None:
    try:
        await asyncio.sleep(max(0.0, delay_sec))
        if session_mgr.is_recording and session_mgr.session_id == session_id:
            try:
                await asyncio.to_thread(session_mgr.stop)
            except Exception as exc:
                _emit_error_event(f"Auto-stop failed: {exc}")
            _reset_lite_state(session_id=None)
            _emit_lite_clear()
    except asyncio.CancelledError:
        return


def _schedule_lite_auto_stop(session_id: str) -> None:
    global LITE_AUTOSTOP_TASK
    if not LITE_MODE or LITE_AUTO_STOP_SEC <= 0:
        return
    if LITE_AUTOSTOP_TASK and not LITE_AUTOSTOP_TASK.done():
        LITE_AUTOSTOP_TASK.cancel()
    LITE_AUTOSTOP_TASK = asyncio.create_task(_auto_stop_worker(session_id, LITE_AUTO_STOP_SEC))


async def _start_session_with_cfg(cfg: Dict[str, Any]) -> str:
    already_recording = bool(session_mgr.is_recording)
    sid = await asyncio.to_thread(session_mgr.start, output_dir_base=str(OUT_DIR))
    if not sid:
        raise RuntimeError("session start failed")
    if not already_recording:
        try:
            session_dir = OUT_DIR / sid
            _write_session_config(session_dir, cfg)
        except Exception:
            pass
        if LITE_MODE:
            _reset_lite_state(session_id=sid)
            _schedule_lite_auto_stop(sid)
    return sid


async def _lite_auto_recovery_task() -> None:
    fail_count = 0
    backoff = max(0.5, LITE_RECOVERY_BACKOFF_SEC)
    while True:
        await asyncio.sleep(1.0)
        if not LITE_MODE or not LITE_AUTO_START:
            await asyncio.sleep(2.0)
            continue
        if session_mgr.is_recording:
            fail_count = 0
            backoff = max(0.5, LITE_RECOVERY_BACKOFF_SEC)
            await asyncio.sleep(2.0)
            continue
        try:
            cfg = _get_model_cfg()
            await _start_session_with_cfg(cfg)
            fail_count = 0
            backoff = max(0.5, LITE_RECOVERY_BACKOFF_SEC)
            await asyncio.sleep(1.0)
        except Exception as exc:
            fail_count += 1
            if fail_count >= max(1, LITE_RECOVERY_MAX_FAILS):
                _emit_error_event(f"Auto-start failed: {exc}")
                fail_count = 0
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)


async def _lite_heartbeat_task() -> None:
    if LITE_HEARTBEAT_SEC <= 0:
        return
    while True:
        await asyncio.sleep(LITE_HEARTBEAT_SEC)
        if not LITE_MODE or not manager.active:
            continue
        payload = {"type": "lite_heartbeat", "ts": time.time()}
        try:
            _enqueue_event(payload)
        except Exception:
            pass


def _env_summary() -> Dict[str, Any]:
    openai_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_ENDPOINT") or "https://api.openai.com"
    openai_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    openai_asr_model = os.getenv("OPENAI_ASR_MODEL") or "whisper-1"
    return {
        "openai_base_url": openai_base,
        "openai_model": openai_model,
        "openai_asr_model": openai_asr_model,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")),
        "has_dashscope_key": bool(os.getenv("DASH_SCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
        "has_xfyun": bool(os.getenv("XFYUN_APP_ID") and os.getenv("XFYUN_SECRET_KEY")),
    }


def _default_model_cfg() -> Dict[str, Any]:
    env = _env_summary()
    has_openai = bool(env.get("has_openai_key"))
    llm_lang = normalize_language(os.getenv("REPORT_LANGUAGE") or os.getenv("SUMMARY_LANGUAGE") or "zh", default="zh", allow_auto=True)
    asr_lang = normalize_language(os.getenv("ASR_LANGUAGE") or os.getenv("OPENAI_ASR_LANGUAGE") or "auto", default="auto", allow_auto=True)
    return {
        "mode": "online" if has_openai else "offline",
        "llm": {
            "enabled": bool(has_openai),
            "provider": "openai_compat",
            "model": str(env.get("openai_model") or "gpt-4o-mini"),
            "base_url": str(env.get("openai_base_url") or "https://api.openai.com"),
            # Optional override (kept in memory only; never persisted to session outputs).
            "api_key": "",
            "language": llm_lang,
        },
        "asr": {
            "provider": "openai_compat" if has_openai else "none",
            "model": str(env.get("openai_asr_model") or "whisper-1"),
            "use_independent": False,
            "base_url": "",
            "api_key": "",
            "language": asr_lang,
        },
    }


def _sanitize_model_cfg(cfg: Any) -> Dict[str, Any]:
    base = _default_model_cfg()
    if not isinstance(cfg, dict):
        return base
    out = {
        "mode": cfg.get("mode", base["mode"]),
        "llm": dict(base["llm"]),
        "asr": dict(base["asr"]),
    }
    if out["mode"] not in ("online", "offline"):
        out["mode"] = base["mode"]

    llm = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    out["llm"]["enabled"] = bool(llm.get("enabled", out["llm"]["enabled"]))
    provider = str(llm.get("provider", out["llm"]["provider"]) or "").strip()
    if provider not in ("openai_compat", "none"):
        provider = out["llm"]["provider"]
    out["llm"]["provider"] = provider
    model = str(llm.get("model", out["llm"]["model"]) or "").strip()
    if model:
        out["llm"]["model"] = model[:120]
    base_url = str(llm.get("base_url", out["llm"]["base_url"]) or "").strip()
    if base_url:
        out["llm"]["base_url"] = base_url[:300]
    api_key = str(llm.get("api_key", out["llm"].get("api_key", "")) or "").strip()
    if api_key:
        # Don't try to validate format; just keep it bounded.
        out["llm"]["api_key"] = api_key[:500]
    llm_lang = normalize_language(
        llm.get("language", out["llm"].get("language", "zh")),
        default=out["llm"].get("language", "zh"),
        allow_auto=True,
    )
    out["llm"]["language"] = llm_lang

    asr = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
    asr_provider = str(asr.get("provider", out["asr"]["provider"]) or "").strip()
    if asr_provider not in ("openai_compat", "dashscope", "xfyun_raasr", "none"):
        asr_provider = out["asr"]["provider"]
    out["asr"]["provider"] = asr_provider
    asr_model = str(asr.get("model", out["asr"]["model"]) or "").strip()
    if asr_model:
        out["asr"]["model"] = asr_model[:120]
    # New ASR independent settings
    out["asr"]["use_independent"] = bool(asr.get("use_independent", False))
    asr_base_url = str(asr.get("base_url", "") or "").strip()
    if asr_base_url:
        out["asr"]["base_url"] = asr_base_url[:300]
    else:
        out["asr"]["base_url"] = ""
    asr_api_key = str(asr.get("api_key", "") or "").strip()
    if asr_api_key:
        out["asr"]["api_key"] = asr_api_key[:500]
    else:
        out["asr"]["api_key"] = ""
    asr_lang = normalize_language(
        asr.get("language", out["asr"].get("language", "auto")),
        default=out["asr"].get("language", "auto"),
        allow_auto=True,
    )
    out["asr"]["language"] = asr_lang
    return out


def _get_model_cfg() -> Dict[str, Any]:
    global _MODEL_CFG
    if not _MODEL_CFG:
        _MODEL_CFG = _default_model_cfg()
    return dict(_MODEL_CFG)


def _set_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _MODEL_CFG
    _MODEL_CFG = _sanitize_model_cfg(cfg)
    return dict(_MODEL_CFG)


# ===== WebDAV Configuration =====
_WEBDAV_CFG: Dict[str, Any] = {}
_WEBDAV_CONFIG_PATH = Path(__file__).resolve().parent.parent / "webdav_config.json"


def _default_webdav_cfg() -> Dict[str, Any]:
    return {
        "enabled": False,
        "url": "",
        "username": "",
        "password": "",
        "remote_path": "/classroom_focus",
        "upload_video": True,
        "upload_audio": True,
        "upload_stats": True,
        "upload_transcript": True,
        "upload_all": False,
        "auto_upload": False,  # Auto upload after recording stops
    }


def _get_webdav_cfg() -> Dict[str, Any]:
    global _WEBDAV_CFG
    if not _WEBDAV_CFG:
        _WEBDAV_CFG = _load_webdav_cfg()
    return dict(_WEBDAV_CFG)


def _set_webdav_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _WEBDAV_CFG
    base = _default_webdav_cfg()
    if not isinstance(cfg, dict):
        cfg = {}
    _WEBDAV_CFG = {
        "enabled": bool(cfg.get("enabled", base["enabled"])),
        "url": str(cfg.get("url", base["url"])).strip(),
        "username": str(cfg.get("username", base["username"])).strip(),
        "password": str(cfg.get("password", base["password"])),
        "remote_path": str(cfg.get("remote_path", base["remote_path"])).strip() or "/classroom_focus",
        "upload_video": bool(cfg.get("upload_video", base["upload_video"])),
        "upload_audio": bool(cfg.get("upload_audio", base["upload_audio"])),
        "upload_stats": bool(cfg.get("upload_stats", base["upload_stats"])),
        "upload_transcript": bool(cfg.get("upload_transcript", base["upload_transcript"])),
        "upload_all": bool(cfg.get("upload_all", base["upload_all"])),
        "auto_upload": bool(cfg.get("auto_upload", base["auto_upload"])),
    }
    _save_webdav_cfg(_WEBDAV_CFG)
    return dict(_WEBDAV_CFG)


def _load_webdav_cfg() -> Dict[str, Any]:
    try:
        if _WEBDAV_CONFIG_PATH.exists():
            with open(_WEBDAV_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    base = _default_webdav_cfg()
                    for k in base:
                        if k not in data:
                            data[k] = base[k]
                    return data
    except Exception:
        pass
    return _default_webdav_cfg()


def _save_webdav_cfg(cfg: Dict[str, Any]) -> bool:
    try:
        with open(_WEBDAV_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _redact_webdav_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Redact password for API response."""
    out = dict(cfg)
    if out.get("password"):
        out["password"] = "***"
    return out


def _openai_client_from_cfg(llm_cfg: Dict[str, Any]) -> Optional[OpenAICompat]:
    api_key = str(llm_cfg.get("api_key") or "").strip() or (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "")
    if not api_key:
        return None
    base_url = str(llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com")
    model = str(llm_cfg.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    try:
        timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    except Exception:
        timeout = 60.0
    from tools.openai_compat import OpenAICompatConfig

    return OpenAICompat(OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model, timeout_sec=timeout))


def _openai_client_for_asr(model_cfg: Dict[str, Any]) -> Optional[OpenAICompat]:
    """Get an OpenAI client for ASR, using independent settings if configured."""
    asr_cfg = model_cfg.get("asr") if isinstance(model_cfg.get("asr"), dict) else {}
    llm_cfg = model_cfg.get("llm") if isinstance(model_cfg.get("llm"), dict) else {}
    use_independent = bool(asr_cfg.get("use_independent", False))
    
    if use_independent:
        # Use ASR-specific settings
        asr_api_key = str(asr_cfg.get("api_key") or "").strip()
        asr_base_url = str(asr_cfg.get("base_url") or "").strip()
        # Fallback to LLM settings if ASR settings are empty
        api_key = asr_api_key or str(llm_cfg.get("api_key") or "").strip() or (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "")
        base_url = asr_base_url or str(llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com")
    else:
        # Use LLM settings
        api_key = str(llm_cfg.get("api_key") or "").strip() or (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "")
        base_url = str(llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com")
    
    if not api_key:
        return None
    
    model = str(asr_cfg.get("model") or os.getenv("OPENAI_ASR_MODEL") or "whisper-1")
    try:
        timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    except Exception:
        timeout = 60.0
    from tools.openai_compat import OpenAICompatConfig

    return OpenAICompat(OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model, timeout_sec=timeout))


def _redact_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove secrets before persisting to disk."""
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg)
    llm = out.get("llm") if isinstance(out.get("llm"), dict) else {}
    llm2 = dict(llm)
    if "api_key" in llm2:
        llm2["api_key"] = ""
    out["llm"] = llm2
    # Also redact ASR api_key
    asr = out.get("asr") if isinstance(out.get("asr"), dict) else {}
    asr2 = dict(asr)
    if "api_key" in asr2:
        asr2["api_key"] = ""
    out["asr"] = asr2
    return out


def _write_session_config(session_dir: Path, cfg: Dict[str, Any]) -> None:
    try:
        meta = {}
        try:
            llm = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
            asr = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
            mode = str(cfg.get("mode") or "").strip()
            if mode in ("online", "offline"):
                meta["mode"] = mode
            if llm.get("language") is not None:
                meta["llm_language"] = normalize_language(llm.get("language"), default="zh", allow_auto=True)
            if asr.get("language") is not None:
                meta["asr_language"] = normalize_language(asr.get("language"), default="auto", allow_auto=True)
        except Exception:
            meta = {}
        with open(session_dir / "session_config.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_config": _redact_model_cfg(cfg),
                    "env_summary": _env_summary(),
                    "meta": meta,
                    "ts": time.time(),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


def _read_session_config(session_dir: Path) -> Dict[str, Any]:
    try:
        p = session_dir / "session_config.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            cfg = j.get("model_config") if isinstance(j, dict) else None
            # session_config is redacted, merge any in-memory secrets (e.g. api_key) back in.
            loaded = _sanitize_model_cfg(cfg)
            cur = _get_model_cfg()
            try:
                meta = j.get("meta") if isinstance(j, dict) else None
                loaded = merge_session_meta(loaded, meta)
                loaded = merge_redacted_model_cfg(loaded, cur)
            except Exception:
                pass
            return loaded
    except Exception:
        pass
    return _get_model_cfg()


def _make_silence_wav(path: Path, seconds: float = 0.6, sr: int = 16000) -> None:
    import wave

    seconds = max(0.2, float(seconds))
    sr = max(8000, int(sr))
    n = int(seconds * sr)
    pcm = b"\x00\x00" * n
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def _hints_for_openai_compat(llm_cfg: Dict[str, Any], *, context: str, err: str = "") -> List[str]:
    hints: List[str] = []
    base_url = str(llm_cfg.get("base_url") or "").strip()
    api_key = str(llm_cfg.get("api_key") or "").strip() or (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "")
    if not base_url:
        hints.append("请设置 Base URL（示例：`https://api.openai.com` 或兼容服务的地址，如 `https://openrouter.ai/api/v1`）。")
    if not api_key:
        hints.append("请设置 API Key（在模型中心填写，或设置环境变量 `OPENAI_API_KEY` / `OPENAI_KEY`）。")

    e = str(err or "").strip()
    el = e.lower()
    if e:
        if "401" in el or "unauthorized" in el or "invalid_api_key" in el:
            hints.append("鉴权失败（401）：API Key 不正确或无权限。")
        if "403" in el or "forbidden" in el:
            hints.append("权限不足（403）：检查 Key 权限/额度/是否需要开通对应模型。")
        if "404" in el:
            hints.append("接口不存在（404）：检查 Base URL 是否正确（有的服务需要带 `/v1`，有的不能重复带）。")
        if "timed out" in el or "timeout" in el:
            hints.append("请求超时：检查网络/代理/服务是否可达，或调大 `OPENAI_TIMEOUT_SEC`。")
        if "name or service not known" in el or "failed to establish a new connection" in el or "connection" in el:
            hints.append("连接失败：检查 Base URL 域名、网络、DNS、代理设置。")
        if "certificate" in el or "ssl" in el:
            hints.append("TLS/证书错误：检查系统证书、抓包代理、或使用可信 HTTPS 入口。")

    if context == "asr":
        hints.append("若该服务不支持 `/audio/transcriptions`，请在 ASR 里切换到 `dashscope` / `xfyun_raasr` 或 `none`。")
    return hints


def _check_models(cfg: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
    mode = cfg.get("mode", "offline")
    out: Dict[str, Any] = {"mode": mode, "llm": {}, "asr": {}}
    env = _env_summary()
    out["env"] = env

    if mode == "offline":
        out["llm"] = {"ok": True, "skipped": True, "reason": "offline mode"}
        out["asr"] = {"ok": True, "skipped": True, "reason": "offline mode"}
        return out

    # LLM check (OpenAI compatible)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    llm_enabled = bool(llm_cfg.get("enabled"))
    llm_provider = str(llm_cfg.get("provider") or "none")
    if not llm_enabled or llm_provider == "none":
        out["llm"] = {"ok": True, "skipped": True, "reason": "llm disabled"}
    else:
        t0 = time.time()
        client = _openai_client_from_cfg(llm_cfg)
        if not client:
            out["llm"] = {
                "ok": False,
                "error": "API key not set",
                "hints": _hints_for_openai_compat(llm_cfg, context="llm", err="API key not set"),
            }
        else:
            try:
                txt = client.generate_text(messages=[{"role": "user", "content": "ping"}], max_tokens=8, temperature=0.0)
                out["llm"] = {
                    "ok": True,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "sample": (txt or "")[:60],
                }
            except Exception as exc:
                out["llm"] = {
                    "ok": False,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "error": str(exc),
                    "hints": _hints_for_openai_compat(llm_cfg, context="llm", err=str(exc)),
                }

    # ASR check
    asr_cfg = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
    asr_provider = str(asr_cfg.get("provider") or "none")
    asr_model = str(asr_cfg.get("model") or "")
    if asr_provider == "none":
        out["asr"] = {"ok": True, "skipped": True, "reason": "asr disabled"}
        return out

    # create a tiny wav for check
    tmp_dir = Path(tempfile.mkdtemp(prefix="model_check_", dir=str(OUT_DIR)))
    wav = tmp_dir / "silence.wav"
    try:
        _make_silence_wav(wav)
        if asr_provider == "openai_compat":
            t0 = time.time()
            client = _openai_client_for_asr(cfg)
            # Build effective config for hints (consider independent ASR settings)
            asr_cfg_for_hints = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
            llm_cfg_for_hints = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
            use_indep = bool(asr_cfg_for_hints.get("use_independent", False))
            effective_hints_cfg = {
                "base_url": (asr_cfg_for_hints.get("base_url") if use_indep else None) or llm_cfg_for_hints.get("base_url"),
                "api_key": (asr_cfg_for_hints.get("api_key") if use_indep else None) or llm_cfg_for_hints.get("api_key"),
            }
            if not client:
                out["asr"] = {
                    "ok": False,
                    "error": "API key not set",
                    "hints": _hints_for_openai_compat(effective_hints_cfg, context="asr", err="API key not set"),
                }
            else:
                try:
                    resp = client.transcribe_audio(str(wav), model=asr_model or None, response_format="verbose_json")
                    ok = isinstance(resp, dict) and ("text" in resp or "segments" in resp)
                    out["asr"] = {
                        "ok": bool(ok),
                        "latency_ms": int((time.time() - t0) * 1000),
                    }
                except Exception as exc:
                    out["asr"] = {
                        "ok": False,
                        "latency_ms": int((time.time() - t0) * 1000),
                        "error": str(exc),
                        "hints": _hints_for_openai_compat(effective_hints_cfg, context="asr", err=str(exc)),
                    }
        elif asr_provider == "dashscope":
            # Quick: validate key+deps, Deep: run a tiny file transcription.
            key = os.getenv("DASH_SCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
            if not key:
                out["asr"] = {
                    "ok": False,
                    "error": "DASH_SCOPE_API_KEY not set",
                    "hints": ["请设置环境变量 `DASH_SCOPE_API_KEY`（或 `DASHSCOPE_API_KEY`）。"],
                }
            else:
                if not deep:
                    out["asr"] = {"ok": True, "skipped": True, "reason": "dashscope key present (deep check disabled)"}
                else:
                    t0 = time.time()
                    try:
                        segs = dashscope_transcribe_wav_to_segments(str(wav), DashScopeOfflineConfig(api_key=key, model=asr_model or "fun-asr-realtime"))
                        out["asr"] = {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "segments": len(segs)}
                    except Exception as exc:
                        hints = []
                        if "dashscope is required" in str(exc).lower():
                            hints.append("请安装可选依赖：`pip install -r requirements-optional.txt`（包含 dashscope）。")
                        out["asr"] = {
                            "ok": False,
                            "latency_ms": int((time.time() - t0) * 1000),
                            "error": str(exc),
                            "hints": hints,
                        }
        elif asr_provider == "xfyun_raasr":
            app_id = os.getenv("XFYUN_APP_ID") or ""
            secret = os.getenv("XFYUN_SECRET_KEY") or ""
            if not app_id or not secret:
                out["asr"] = {
                    "ok": False,
                    "error": "XFYUN_APP_ID/XFYUN_SECRET_KEY not set",
                    "hints": ["请设置 `XFYUN_APP_ID` 和 `XFYUN_SECRET_KEY`（讯飞 RaaSR）。"],
                }
            else:
                if not deep:
                    out["asr"] = {"ok": True, "skipped": True, "reason": "xfyun keys present (deep check disabled)"}
                else:
                    t0 = time.time()
                    try:
                        segs, meta = xfyun_raasr_transcribe_wav(
                            str(wav),
                            XfyunRaasrConfig(app_id=app_id, secret_key=secret),
                        )
                        out["asr"] = {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "segments": len(segs), "task_id": meta.get("task_id")}
                    except Exception as exc:
                        out["asr"] = {
                            "ok": False,
                            "latency_ms": int((time.time() - t0) * 1000),
                            "error": str(exc),
                            "hints": ["请检查讯飞 Key 是否正确、账户余额、以及 `XFYUN_RAASR_HOST` 是否可达。"],
                        }
        else:
            out["asr"] = {"ok": False, "error": f"unknown asr provider: {asr_provider}"}
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return out


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        _mark_client_activity()
        _restore_preview_defaults()
        if LITE_MODE:
            try:
                await ws.send_text(json.dumps(_lite_snapshot_payload()))
            except Exception:
                pass

    def disconnect(self, ws: WebSocket):
        try:
            self.active.remove(ws)
        except ValueError:
            pass
        if not self.active:
            _mark_client_activity()

    async def broadcast(self, message: str):
        coros = []
        targets = list(self.active)
        for ws in targets:
            coros.append(ws.send_text(message))
        if not coros:
            return
        results = await asyncio.gather(*coros, return_exceptions=True)
        for ws, res in zip(targets, results):
            if isinstance(res, Exception):
                try:
                    self.active.remove(ws)
                except ValueError:
                    pass


manager = ConnectionManager()


def _restore_preview_defaults() -> None:
    global PREVIEW_THROTTLED
    session_mgr.set_preview(enabled=PREVIEW_DEFAULT_ENABLED, interval_sec=PREVIEW_DEFAULT_INTERVAL)
    session_mgr.set_preview_thumbs(
        enabled=PREVIEW_DEFAULT_THUMBS,
        size=PREVIEW_DEFAULT_THUMB_CFG["size"],
        quality=PREVIEW_DEFAULT_THUMB_CFG["quality"],
        pad=PREVIEW_DEFAULT_THUMB_CFG["pad"],
        refresh_sec=PREVIEW_DEFAULT_THUMB_CFG["refresh_sec"],
        refresh_frames=PREVIEW_DEFAULT_THUMB_CFG["refresh_frames"],
    )
    PREVIEW_THROTTLED = False


def _apply_idle_throttle() -> None:
    global PREVIEW_THROTTLED
    if PREVIEW_THROTTLED:
        return
    interval = float(WEB_IDLE_PREVIEW_INTERVAL_SEC or 0.0)
    enabled = interval > 0.0
    session_mgr.set_preview(enabled=enabled, interval_sec=interval if enabled else None)
    if not enabled:
        session_mgr.set_preview_thumbs(enabled=False)
    PREVIEW_THROTTLED = True


def _is_idle() -> bool:
    if float(WEB_IDLE_THROTTLE_SEC or 0.0) <= 0.0:
        return False
    if manager.active:
        return False
    return (time.time() - float(LAST_CLIENT_TS or 0.0)) >= float(WEB_IDLE_THROTTLE_SEC)


async def broadcaster_task():
    """Background task that batches events and broadcasts at a fixed interval."""
    batch_interval = 0.5 if LITE_MODE else 0.1
    while True:
        batch = []
        try:
            # wait for first item
            item = await asyncio.wait_for(event_queue.get(), timeout=batch_interval)
            resolved = _resolve_coalesced_item(item)
            if resolved is not None:
                batch.append(resolved)
        except asyncio.TimeoutError:
            # nothing arrived in interval
            pass

        # drain queue without blocking
        while True:
            try:
                item = event_queue.get_nowait()
                resolved = _resolve_coalesced_item(item)
                if resolved is not None:
                    batch.append(resolved)
            except asyncio.QueueEmpty:
                break

        if batch:
            stats['last_batch_size'] = len(batch)
            stats['batches_broadcast'] += 1
            payload = json.dumps({"type": "batch", "events": batch})
            await manager.broadcast(payload)

        await asyncio.sleep(batch_interval)


async def stats_logger():
    while True:
        try:
            print(
                f"[web_viz_server] stats received={stats['received']} "
                f"batches_broadcast={stats['batches_broadcast']} "
                f"last_batch_size={stats['last_batch_size']} "
                f"dropped_events={stats['dropped_events']} "
                f"coalesced_updates={stats['coalesced_updates']}"
            )
        except Exception:
            pass
        await asyncio.sleep(WEB_IDLE_STATS_INTERVAL_SEC if _is_idle() else 2.0)


async def _idle_throttle_task() -> None:
    if float(WEB_IDLE_THROTTLE_SEC or 0.0) <= 0.0:
        return
    while True:
        await asyncio.sleep(1.0)
        if _is_idle():
            _apply_idle_throttle()
        elif PREVIEW_THROTTLED:
            _restore_preview_defaults()


# Global loop reference
main_loop = None


@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_running_loop()

    # Configure session manager callback
    def callback_wrapper(data, img_bytes):
        if not main_loop:
            return
        if LITE_MODE:
            if isinstance(data, dict) and data.get("type") == "error":
                if manager.active:
                    _queue_event_threadsafe(data)
                return
            payload = _handle_lite_frame(data, allow_emit=bool(manager.active), img_bytes=img_bytes)
            if payload and manager.active:
                _queue_event_threadsafe(payload)
            return
        if not manager.active:
            return
        if img_bytes:
            data['image_base64'] = base64.b64encode(img_bytes).decode('utf-8')
        _queue_event_threadsafe(data)

    session_mgr.set_callback(callback_wrapper)

    # start background tasks
    asyncio.create_task(broadcaster_task())
    asyncio.create_task(stats_logger())
    asyncio.create_task(_idle_throttle_task())
    if LITE_MODE:
        asyncio.create_task(_lite_heartbeat_task())
        asyncio.create_task(_lite_auto_recovery_task())


@app.get("/")
async def root():
    return RedirectResponse(url="/static/viz_lite.html" if LITE_MODE else "/static/viz.html")


@app.get("/lite")
async def lite_root():
    return RedirectResponse(url="/static/viz_lite.html")


@app.get("/api/ui/config")
async def ui_config():
    return JSONResponse({
        "ok": True,
        "ui_mode": UI_MODE,
        "lite": {
            "enabled": LITE_MODE,
            "status_interval_sec": LITE_STATUS_INTERVAL_SEC,
            "auto_start": LITE_AUTO_START,
            "auto_report": LITE_AUTO_REPORT,
            "disable_video_write": LITE_DISABLE_VIDEO_WRITE,
            "disable_audio": LITE_DISABLE_AUDIO,
            "write_faces": LITE_WRITE_FACES,
            "faces_sample_sec": LITE_FACES_SAMPLE_SEC,
            "auto_stop_sec": LITE_AUTO_STOP_SEC,
            "auto_upload": LITE_AUTO_UPLOAD,
            "adaptive_scheduler": LITE_ADAPTIVE_SCHEDULER,
            "heartbeat_sec": LITE_HEARTBEAT_SEC,
            "thumbnails": LITE_THUMBNAILS,
            "thumb_interval_sec": LITE_THUMB_INTERVAL_SEC,
            "thumb_size": LITE_THUMB_SIZE,
            "thumb_refresh_sec": LITE_THUMB_REFRESH_SEC,
            "thumb_refresh_frames": LITE_THUMB_REFRESH_FRAMES,
        },
    })


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions with display names. Returns fresh data without caching."""
    import asyncio
    import concurrent.futures
    
    def _read_session_info(p):
        """Synchronous helper to read session info from disk."""
        sid = p.name
        try:
            st = p.stat()
            stats_path = p / "stats.json"
            transcript_path = p / "transcript.txt"
            video_path = p / "session.mp4"
            meta_path = p / "metadata.json"
            
            display_name = None
            try:
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        display_name = meta.get("display_name")
            except Exception:
                pass
            
            return {
                "session_id": sid,
                "display_name": display_name,
                "mtime": float(getattr(st, "st_mtime", 0.0)),
                "has_stats": stats_path.exists(),
                "has_transcript": transcript_path.exists(),
                "has_video": video_path.exists() or any(p.glob("*.mp4")),
            }
        except Exception:
            return None
    
    base = OUT_DIR
    sessions = []
    try:
        dirs = [p for p in base.iterdir() if p.is_dir()]
        # Performance: Use thread pool to read session info in parallel
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = await asyncio.gather(*[
                loop.run_in_executor(pool, _read_session_info, p) 
                for p in dirs
            ])
        sessions = [r for r in results if r is not None]
    except Exception:
        sessions = []
    sessions.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
    return JSONResponse(
        {"ok": True, "sessions": sessions},
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/device/profile")
async def get_device_profile():
    """Return the detected device profile and available profiles."""
    return JSONResponse({
        "ok": True,
        "current": DEVICE_PROFILE.to_dict(),
        "available": list_profiles(),
    })


@app.post("/api/session/start")
async def start_session(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        cfg = _sanitize_model_cfg((body or {}).get("model_config") or (body or {}).get("config") or _get_model_cfg())
        # Persist as the current default so the UI/session status stays consistent.
        cfg = _set_model_cfg(cfg)

        sid = await _start_session_with_cfg(cfg)
        return {"ok": True, "session_id": sid, "model_config": cfg}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/session/stop")
async def stop_session(background_tasks: BackgroundTasks):
    try:
        path = session_mgr.stop()
        warn = getattr(session_mgr, "video_error", None) or None
        session_id = Path(path).name if path else None
        if LITE_AUTOSTOP_TASK and not LITE_AUTOSTOP_TASK.done():
            LITE_AUTOSTOP_TASK.cancel()
        if LITE_MODE:
            _reset_lite_state(session_id=None)
            _emit_lite_clear()
        
        # Check if auto-upload is enabled
        webdav_cfg = _get_webdav_cfg()
        should_auto_upload = (
            webdav_cfg.get("enabled")
            and webdav_cfg.get("auto_upload")
            and session_id
        )
        
        if should_auto_upload:
            # Schedule background upload
            def _do_webdav_upload(sid: str):
                try:
                    from sync.webdav_client import WebDAVConfig, WebDAVClient
                    config = WebDAVConfig.from_dict(webdav_cfg)
                    if config.is_valid():
                        session_dir = OUT_DIR / sid
                        if session_dir.exists():
                            client = WebDAVClient(config)
                            client.upload_session(str(session_dir), sid)
                except Exception as e:
                    print(f"[WebDAV] Auto-upload failed: {e}")
            
            background_tasks.add_task(_do_webdav_upload, session_id)
        
        return {"ok": True, "path": path, "warning": warn, "auto_upload_scheduled": should_auto_upload}
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


def _estimate_id_switch_stats(
    faces: Optional[List[Dict[str, Any]]] = None,
    faces_path: Optional[Path] = None,
) -> Dict[str, Any]:
    out = {"id_switches": 0, "id_switch_matches": 0, "id_switch_rate": 0.0}
    if faces is None:
        if faces_path is None or not faces_path.exists():
            return out

    try:
        max_dist = float(os.getenv("ID_SWITCH_MAX_NORM_DIST", "0.12"))
    except Exception:
        max_dist = 0.12

    def _center(rec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        c = rec.get("center")
        if isinstance(c, list) and len(c) >= 2:
            try:
                return float(c[0]), float(c[1])
            except Exception:
                return None
        bbox = rec.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                nx, ny, nw, nh = [float(b) for b in bbox[:4]]
            except Exception:
                return None
            return nx + nw / 2.0, ny + nh / 2.0
        return None

    def _track_id(rec: Dict[str, Any]) -> Optional[str]:
        tid = rec.get("track_id")
        if tid is None:
            tid = rec.get("student_id")
        if tid is None:
            return None
        return str(tid)

    def _flush(prev: List[Dict[str, Any]], cur: List[Dict[str, Any]]) -> None:
        if not prev or not cur:
            return
        candidates: List[Tuple[float, int, int]] = []
        for ci, c in enumerate(cur):
            cc = _center(c)
            if cc is None:
                continue
            for pi, p in enumerate(prev):
                pc = _center(p)
                if pc is None:
                    continue
                d = float((cc[0] - pc[0]) ** 2 + (cc[1] - pc[1]) ** 2) ** 0.5
                if d <= max_dist:
                    candidates.append((d, ci, pi))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[0])
        used_cur = set()
        used_prev = set()
        for _, ci, pi in candidates:
            if ci in used_cur or pi in used_prev:
                continue
            used_cur.add(ci)
            used_prev.add(pi)
            cur_id = _track_id(cur[ci])
            prev_id = _track_id(prev[pi])
            if cur_id is None or prev_id is None:
                continue
            out["id_switch_matches"] += 1
            if cur_id != prev_id:
                out["id_switches"] += 1

    prev_dets: List[Dict[str, Any]] = []
    cur_frame = None
    cur_dets: List[Dict[str, Any]] = []
    try:
        if faces is None:
            faces = []
            with open(faces_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        faces.append(rec)
        for rec in faces:
            if not isinstance(rec, dict):
                continue
            frame = rec.get("frame")
            if frame is None:
                try:
                    frame = int(float(rec.get("ts", 0.0)) * 10.0)
                except Exception:
                    frame = None
            if cur_frame is None:
                cur_frame = frame
            if frame != cur_frame:
                _flush(prev_dets, cur_dets)
                prev_dets = cur_dets
                cur_dets = []
                cur_frame = frame
            cur_dets.append(rec)
        _flush(prev_dets, cur_dets)
    except Exception:
        return out

    matches = float(out["id_switch_matches"])
    if matches > 0:
        out["id_switch_rate"] = float(out["id_switches"]) / matches
    return out


def _summarize_text(
    text: str,
    max_points: int = 6,
    llm: Optional[OpenAICompat] = None,
    allow_llm: bool = True,
    language: str = "zh",
):
    """Summarize text into knowledge points.

    Uses an OpenAI-compatible LLM when configured, otherwise falls back to
    lightweight extraction.
    """
    if not text or text.strip() == "":
        return []
    # Keep LLM prompt bounded for long intervals.
    max_chars = int(os.getenv("INTERVAL_LLM_MAX_CHARS", "6000"))
    if len(text) > max_chars:
        half = max(500, max_chars // 2)
        text = text[:half].rstrip() + "\n...\n" + text[-half:].lstrip()
    if allow_llm and llm is None:
        llm = OpenAICompat.from_env()
    if allow_llm and llm:
        try:
            lang_hint = llm_language_hint(normalize_language(language, default="zh", allow_auto=True))
            messages = [
                {"role": "system", "content": f"你从课堂讲解中提炼知识点。{lang_hint}只输出 JSON 数组（字符串数组），不要输出任何额外文字。"},
                {"role": "user", "content": f"从下面转录内容中提炼最多 {int(max_points)} 条知识点（短语化、去重）：\n\n{text}"},
            ]
            pts = llm.generate_json_array(messages=messages, max_tokens=500, temperature=0.2)
            out = []
            for p in pts:
                if not p:
                    continue
                s = str(p).strip()
                if s:
                    out.append(s)
            return out[:max_points]
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


def _chunk_asr_segments(asr_segments: List[dict], chunk_secs: float) -> List[dict]:
    """Group ASR segments into coarse time buckets for lesson topic summarization."""
    chunk_secs = max(30.0, float(chunk_secs))
    chunks = []
    cur = None
    for seg in asr_segments:
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": start, "end": end, "texts": [text]}
            continue
        if (end - cur["start"]) > chunk_secs and cur["texts"]:
            chunks.append({"start": float(cur["start"]), "end": float(cur["end"]), "text": "\n".join(cur["texts"]).strip()})
            cur = {"start": start, "end": end, "texts": [text]}
            continue
        cur["end"] = max(float(cur["end"]), end)
        cur["texts"].append(text)
    if cur and cur.get("texts"):
        chunks.append({"start": float(cur["start"]), "end": float(cur["end"]), "text": "\n".join(cur["texts"]).strip()})
    return chunks


def _llm_summarize_lesson(
    asr_segments: List[dict],
    llm: Optional[OpenAICompat] = None,
    allow_llm: bool = True,
    language: str = "zh",
) -> dict:
    """Summarize the whole lesson and produce a timeline of topics."""
    if not allow_llm:
        return {}
    if llm is None:
        llm = OpenAICompat.from_env()
    if not llm:
        return {}
    lang_hint = llm_language_hint(normalize_language(language, default="zh", allow_auto=True))
    chunk_secs = float(os.getenv("SUMMARY_CHUNK_SECONDS", "180"))
    chunks = _chunk_asr_segments(asr_segments, chunk_secs=chunk_secs)
    if not chunks:
        return {}

    timeline = []
    for ch in chunks:
        messages = [
            {"role": "system", "content": f"你要总结课堂讲解内容。{lang_hint}只输出 JSON 对象，不要输出任何额外文字。"},
            {
                "role": "user",
                "content": (
                    "根据下面这一段课堂转录，提取：\n"
                    "- topic：该时间段主题（短标题）\n"
                    "- summary：2-3 句话总结\n"
                    "- key_points：3-6 条知识点（短语化，去重）\n\n"
                    "只返回 JSON，字段为：topic, summary, key_points。\n\n"
                    f"[时间 {ch['start']:.2f}s - {ch['end']:.2f}s]\n{ch['text']}"
                ),
            },
        ]
        try:
            obj = llm.generate_json_object(messages=messages, max_tokens=700, temperature=0.2)
        except Exception:
            obj = {}
        timeline.append({
            "start": float(ch["start"]),
            "end": float(ch["end"]),
            "topic": str(obj.get("topic", "")).strip() if isinstance(obj, dict) else "",
            "summary": str(obj.get("summary", "")).strip() if isinstance(obj, dict) else "",
            "key_points": [str(x).strip() for x in (obj.get("key_points") or [])] if isinstance(obj, dict) else [],
        })

    # Merge timeline summaries into an overall lesson summary.
    compact = []
    for t in timeline:
        compact.append({
            "start": t["start"],
            "end": t["end"],
            "topic": t["topic"],
            "key_points": t["key_points"][:6],
            "summary": t["summary"],
        })
    messages = [
        {"role": "system", "content": f"你要生成用于课堂专注度分析的课程结构化总结。{lang_hint}只输出 JSON 对象，不要输出任何额外文字。"},
        {
            "role": "user",
            "content": (
                "根据下面按时间分段的总结，生成整节课的总结。\n"
                "只返回 JSON，字段为：\n"
                "- title：课程标题\n"
                "- overview：3-6 句概览\n"
                "- key_points：6-12 条关键要点（短语化）\n"
                "- outline：大纲（章节标题数组）\n"
                f"{lang_hint}\n\n"
                + json.dumps(compact, ensure_ascii=False)
            ),
        },
    ]
    try:
        overall = llm.generate_json_object(messages=messages, max_tokens=900, temperature=0.2)
    except Exception:
        overall = {}

    out = {
        "title": str(overall.get("title", "")).strip() if isinstance(overall, dict) else "",
        "overview": str(overall.get("overview", "")).strip() if isinstance(overall, dict) else "",
        "key_points": [str(x).strip() for x in (overall.get("key_points") or [])] if isinstance(overall, dict) else [],
        "outline": [str(x).strip() for x in (overall.get("outline") or [])] if isinstance(overall, dict) else [],
        "timeline": timeline,
        "language": normalize_language(language, default="zh", allow_auto=True),
    }
    return out


def _audio_sync_offset_sec(session_dir: Path) -> float:
    """Align audio timeline to CV timeline (both relative to session start)."""
    try:
        sync_path = session_dir / "sync.json"
        if sync_path.exists():
            with open(sync_path, "r", encoding="utf-8") as fh:
                sync = json.load(fh)
            return float(sync.get("audio_offset_sec", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def _load_asr_segments(asr_path: Path) -> List[dict]:
    if not asr_path.exists() or asr_path.stat().st_size <= 0:
        return []
    try:
        from analysis.teacher_labeler import load_asr_jsonl
        return load_asr_jsonl(str(asr_path))
    except Exception:
        return _read_jsonl(asr_path)


def _write_asr_segments(asr_path: Path, segments: List[dict], language: Optional[str] = None) -> None:
    try:
        lang = str(language or "").strip()
    except Exception:
        lang = ""
    try:
        with open(asr_path, "w", encoding="utf-8") as fh:
            for s in segments:
                if not isinstance(s, dict):
                    continue
                if lang and "language" not in s:
                    item = dict(s)
                    item["language"] = lang
                else:
                    item = s
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _transcribe_openai_audio_to_segments(
    session_dir: Path,
    job_id: str,
    client: OpenAICompat,
    asr_model: Optional[str],
    language: Optional[str] = None,
) -> List[dict]:
    """Transcribe `temp_audio.wav` via OpenAI-compatible `/audio/transcriptions`."""
    wav_in = session_dir / "temp_audio.wav"
    if not wav_in.exists() or wav_in.stat().st_size <= 0:
        raise RuntimeError(f"audio file missing or empty: {wav_in}")

    import soundfile as sf

    sync_offset = _audio_sync_offset_sec(session_dir)

    # Segment audio to avoid provider size limits.
    chunk_sec = float(os.getenv("ASR_CHUNK_SECONDS", "60"))
    chunk_sec = max(15.0, min(300.0, chunk_sec))
    keep_chunks = os.getenv("KEEP_ASR_CHUNKS", "0").lower() in ("1", "true", "yes")

    tmp_dir = Path(tempfile.mkdtemp(prefix="asr_chunks_", dir=str(session_dir)))
    segments_out: List[dict] = []
    offset = 0.0
    idx = 0
    try:
        with sf.SoundFile(str(wav_in), "r") as f:
            sr = int(f.samplerate)
            ch = int(f.channels)
            frames_per_chunk = int(sr * chunk_sec)
            while True:
                data = f.read(frames_per_chunk, dtype="int16", always_2d=True)
                if data is None or len(data) == 0:
                    break
                dur = float(len(data) / max(1, sr))
                # Ensure mono (most ASR endpoints assume mono).
                if ch > 1:
                    data = data[:, 0:1]
                chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
                sf.write(str(chunk_path), data, sr, subtype="PCM_16")
                _proc_jobs[job_id]["status"] = f"running:asr(openai) chunk {idx} ({offset:.0f}s)"

                try:
                    resp = client.transcribe_audio(
                        str(chunk_path),
                        model=asr_model or None,
                        language=language,
                        response_format="verbose_json",
                    )
                except Exception as exc:
                    raise RuntimeError(f"openai-compatible transcription failed at chunk {idx}: {exc}") from exc

                if isinstance(resp, dict) and isinstance(resp.get("segments"), list):
                    for s in resp["segments"]:
                        if not isinstance(s, dict):
                            continue
                        local_start = float(s.get("start", 0.0))
                        local_end = float(s.get("end", local_start + 1.0))
                        st = local_start + offset + sync_offset
                        en = local_end + offset + sync_offset
                        txt = str(s.get("text", "")).strip()
                        if not txt:
                            continue
                        segments_out.append({"start": st, "end": max(en, st + 0.2), "text": txt, "raw": s})
                else:
                    txt = ""
                    if isinstance(resp, dict) and isinstance(resp.get("text"), str):
                        txt = resp.get("text") or ""
                    elif isinstance(resp, str):
                        txt = resp
                    txt = str(txt).strip()
                    if txt:
                        segments_out.append({"start": offset + sync_offset, "end": offset + sync_offset + max(0.2, dur), "text": txt, "raw": resp})

                offset += dur
                idx += 1
    finally:
        if not keep_chunks:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    segments_out.sort(key=lambda s: float(s.get("start", 0.0)))
    return segments_out


def _transcribe_dashscope_audio_to_segments(session_dir: Path, job_id: str, model: Optional[str]) -> List[dict]:
    wav_in = session_dir / "temp_audio.wav"
    if not wav_in.exists() or wav_in.stat().st_size <= 0:
        raise RuntimeError(f"audio file missing or empty: {wav_in}")
    key = os.getenv("DASH_SCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    if not key:
        raise RuntimeError("DASH_SCOPE_API_KEY not set.")
    _proc_jobs[job_id]["status"] = "running:asr(dashscope)"
    segs = dashscope_transcribe_wav_to_segments(
        str(wav_in),
        DashScopeOfflineConfig(api_key=key, model=model or (os.getenv("ALI_ASR_MODEL") or "fun-asr-realtime")),
    )
    sync_offset = _audio_sync_offset_sec(session_dir)
    for s in segs:
        try:
            s["start"] = float(s.get("start", 0.0)) + sync_offset
            s["end"] = float(s.get("end", s["start"] + 0.2)) + sync_offset
        except Exception:
            pass
    segs.sort(key=lambda s: float(s.get("start", 0.0)))
    return segs


def _transcribe_xfyun_audio_to_segments(session_dir: Path, job_id: str) -> List[dict]:
    wav_in = session_dir / "temp_audio.wav"
    if not wav_in.exists() or wav_in.stat().st_size <= 0:
        raise RuntimeError(f"audio file missing or empty: {wav_in}")
    app_id = os.getenv("XFYUN_APP_ID") or ""
    secret = os.getenv("XFYUN_SECRET_KEY") or ""
    if not app_id or not secret:
        raise RuntimeError("XFYUN_APP_ID/XFYUN_SECRET_KEY not set.")
    _proc_jobs[job_id]["status"] = "running:asr(xfyun)"
    segs, meta = xfyun_raasr_transcribe_wav(
        str(wav_in),
        XfyunRaasrConfig(
            app_id=app_id,
            secret_key=secret,
            host=os.getenv("XFYUN_RAASR_HOST") or "https://raasr.xfyun.cn/v2/api",
        ),
    )
    try:
        with open(session_dir / "asr_xfyun_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
    sync_offset = _audio_sync_offset_sec(session_dir)
    for s in segs:
        try:
            s["start"] = float(s.get("start", 0.0)) + sync_offset
            s["end"] = float(s.get("end", s["start"] + 0.2)) + sync_offset
        except Exception:
            pass
    segs.sort(key=lambda s: float(s.get("start", 0.0)))
    return segs


def _ensure_asr_segments(session_dir: Path, job_id: str, model_cfg: Dict[str, Any], warnings: List[str]) -> List[dict]:
    """Get ASR segments according to config, avoiding re-transcription when possible."""
    asr_path = session_dir / "asr.jsonl"
    existing = _load_asr_segments(asr_path)
    if existing:
        return existing

    mode = str(model_cfg.get("mode") or "offline")
    if mode == "offline":
        return []

    asr_cfg = model_cfg.get("asr") if isinstance(model_cfg.get("asr"), dict) else {}
    llm_cfg = model_cfg.get("llm") if isinstance(model_cfg.get("llm"), dict) else {}
    provider = str(asr_cfg.get("provider") or "none")
    model = str(asr_cfg.get("model") or "").strip() or None
    llm_lang = normalize_language(llm_cfg.get("language", "zh"), default="zh", allow_auto=True)
    asr_lang = resolve_asr_language(asr_cfg.get("language", "auto"), llm_lang)
    asr_lang_param = asr_language_param(asr_lang)

    try:
        if provider == "none":
            return []
        if provider == "openai_compat":
            client = _openai_client_for_asr(model_cfg)
            if not client:
                raise RuntimeError("OPENAI_API_KEY not set.")
            segs = _transcribe_openai_audio_to_segments(session_dir, job_id, client, model, language=asr_lang_param)
        elif provider == "dashscope":
            segs = _transcribe_dashscope_audio_to_segments(session_dir, job_id, model)
        elif provider == "xfyun_raasr":
            segs = _transcribe_xfyun_audio_to_segments(session_dir, job_id)
        else:
            raise RuntimeError(f"unknown ASR provider: {provider}")
    except Exception as exc:
        warnings.append(f"ASR failed ({provider}): {exc}")
        return []

    _write_asr_segments(asr_path, segs, language=asr_lang)
    return segs


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

    # Use stable repo-root based output dir.
    session_dir = OUT_DIR / sid
    if not session_dir.exists():
        return JSONResponse({"ok": False, "error": f"session dir not found: {session_dir}"}, status_code=404)

    job_id = f"job_{int(time.time())}"
    _proc_jobs[job_id] = {"status": "queued", "session_id": sid}

    def _do_work(job_id, session_dir):
        try:
            _proc_jobs[job_id]['status'] = 'running'
            warnings: List[str] = []
            model_cfg = _read_session_config(session_dir)
            llm_cfg = model_cfg.get("llm") if isinstance(model_cfg.get("llm"), dict) else {}
            asr_cfg = model_cfg.get("asr") if isinstance(model_cfg.get("asr"), dict) else {}
            llm_lang = normalize_language(llm_cfg.get("language", "zh"), default="zh", allow_auto=True)
            asr_lang = resolve_asr_language(asr_cfg.get("language", "auto"), llm_lang)
            allow_llm = (
                str(model_cfg.get("mode") or "offline") == "online"
                and bool(llm_cfg.get("enabled"))
                and str(llm_cfg.get("provider") or "none") != "none"
            )
            llm_client: Optional[OpenAICompat] = None
            if allow_llm:
                llm_client = _openai_client_from_cfg(llm_cfg)
                if not llm_client:
                    allow_llm = False
                    warnings.append("LLM disabled: OPENAI_API_KEY not set or invalid.")

            # Find video file (prefer mp4 then avi)
            video_candidates = list(session_dir.glob('session.mp4')) + list(session_dir.glob('*.mp4')) + list(session_dir.glob('temp_video.avi'))
            video_path = video_candidates[0] if video_candidates else None

            # 1) Ensure whole-class audio -> text (ASR)
            _proc_jobs[job_id]["status"] = "running:asr"
            asr_segments = _ensure_asr_segments(session_dir, job_id, model_cfg, warnings)
            transcript_txt = session_dir / "transcript.txt"
            try:
                with open(transcript_txt, "w", encoding="utf-8") as fh:
                    if asr_lang:
                        fh.write(f"[language: {asr_lang}]\n")
                    if asr_segments:
                        for seg in asr_segments:
                            if not isinstance(seg, dict):
                                continue
                            txt = str(seg.get("text", "")).strip()
                            if txt:
                                fh.write(txt + "\n")
                    else:
                        mode = str(model_cfg.get("mode") or "offline")
                        asr_provider = str(((model_cfg.get("asr") or {}) if isinstance(model_cfg.get("asr"), dict) else {}).get("provider") or "none")
                        if mode == "offline":
                            fh.write("[ASR skipped: offline mode]\n")
                        elif asr_provider == "none":
                            fh.write("[ASR disabled]\n")
                        else:
                            fh.write("[ASR failed or produced no text]\n")
            except Exception:
                pass

            # 2) Summarize the whole lesson via OpenAI-compatible LLM (if configured)
            _proc_jobs[job_id]["status"] = "running:lesson_summary"
            lesson_summary = _llm_summarize_lesson(
                asr_segments,
                llm=llm_client,
                allow_llm=allow_llm,
                language=llm_lang,
            )
            lesson_summary_path = session_dir / "lesson_summary.json"
            if lesson_summary:
                try:
                    with open(lesson_summary_path, "w", encoding="utf-8") as fh:
                        json.dump(lesson_summary, fh, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # Parse cv events to build per-student non-awake intervals
            _proc_jobs[job_id]["status"] = "running:cv_intervals"
            events_path = session_dir / 'cv_events.jsonl'
            faces_path = session_dir / 'faces.jsonl'
            events = _read_jsonl(events_path) if events_path.exists() else []
            faces = _read_jsonl(faces_path) if faces_path.exists() else []
            id_switch_stats = _estimate_id_switch_stats(faces=faces)

            per_student = {}

            # Determine session end time for closing open intervals / inferring not-visible gaps.
            session_end_ts = 0.0
            try:
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    try:
                        session_end_ts = max(session_end_ts, float(ev.get("ts", 0.0)))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                for fr in faces:
                    if not isinstance(fr, dict):
                        continue
                    try:
                        session_end_ts = max(session_end_ts, float(fr.get("ts", 0.0)))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                for seg in asr_segments:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        session_end_ts = max(session_end_ts, float(seg.get("end", seg.get("start", 0.0))))
                    except Exception:
                        pass
            except Exception:
                pass

            # Inattentive definition:
            # - DROWSY (eyes closed)
            # - LOOKING_DOWN (clear distraction)
            # - NOT_VISIBLE (eyes not visible; e.g. head down/occluded)
            keep_types = ("DROWSY_START", "DROWSY_END", "LOOKING_DOWN_START", "LOOKING_DOWN_END")
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                typ = ev.get("type")
                if typ not in keep_types:
                    continue
                sid_ev = str(ev.get("student_id", ev.get("track_id", "unknown")))
                try:
                    ts = float(ev.get("ts", 0.0))
                except Exception:
                    ts = 0.0
                if sid_ev not in per_student:
                    per_student[sid_ev] = {}
                if typ.endswith("_START"):
                    per_student[sid_ev].setdefault("open", []).append({"type": typ.replace("_START", ""), "start": ts, "end": None})
                elif typ.endswith("_END"):
                    name = typ.replace("_END", "")
                    opens = per_student[sid_ev].get("open", [])
                    for o in reversed(opens):
                        if o.get("type") == name and o.get("end") is None:
                            o["end"] = ts
                            break

            # finalize raw intervals from CV events
            for sid_ev, info in per_student.items():
                intervals = []
                for o in info.get("open", []):
                    start = float(o.get("start", 0.0))
                    end = o.get("end")
                    if end is None:
                        end = session_end_ts or (start + 1.0)
                    end = float(end)
                    if end <= start:
                        end = start + 0.5
                    intervals.append({"type": str(o.get("type") or "UNKNOWN"), "start": float(start), "end": float(end)})
                info["raw_intervals"] = intervals
                info.pop("open", None)

            # infer NOT_VISIBLE intervals from face gaps
            gap_sec = float(os.getenv("NOT_VISIBLE_GAP_SEC", "1.6"))
            tail_cap = os.getenv("NOT_VISIBLE_TAIL_CAP_SEC", "")
            tail_cap_sec = None
            try:
                if tail_cap != "":
                    tail_cap_sec = float(tail_cap)
                else:
                    tail_cap_sec = 12.0
            except Exception:
                tail_cap_sec = 12.0
            if tail_cap_sec is not None and tail_cap_sec <= 0:
                tail_cap_sec = None

            face_times = {}
            for fr in faces:
                if not isinstance(fr, dict):
                    continue
                sid = str(fr.get("student_id", fr.get("track_id", "unknown")))
                try:
                    tsv = float(fr.get("ts", 0.0))
                except Exception:
                    continue
                face_times.setdefault(sid, []).append(tsv)

            for sid, times in face_times.items():
                nv = infer_not_visible_intervals(times, session_end=session_end_ts, gap_sec=gap_sec, tail_cap_sec=tail_cap_sec)
                if not nv:
                    continue
                info = per_student.setdefault(sid, {})
                info.setdefault("raw_intervals", []).extend(nv)

            # Merge into union inattentive intervals to avoid double-counting overlaps.
            merge_gap = float(os.getenv("INATTENTIVE_MERGE_GAP_SEC", "0.4"))
            for sid_ev in list(per_student.keys()):
                info = per_student[sid_ev]
                raw = info.get("raw_intervals") or []
                merged = merge_inattentive_intervals(raw, join_gap_sec=merge_gap, out_type="INATTENTIVE")
                info["intervals"] = merged
                if not merged:
                    per_student.pop(sid_ev, None)

            # Associate ASR segments to intervals
            _proc_jobs[job_id]["status"] = "running:align_asr"
            lesson_timeline = lesson_summary.get("timeline") if isinstance(lesson_summary, dict) else None
            if not isinstance(lesson_timeline, list):
                lesson_timeline = []
            for sid_ev, info in per_student.items():
                for it in info['intervals']:
                    # gather ASR segments that overlap interval
                    txts = []
                    segs = []
                    for seg in asr_segments:
                        if not isinstance(seg, dict):
                            continue
                        s = float(seg.get('start', seg.get('ts', 0.0)))
                        e = float(seg.get('end', s + float(seg.get('duration', 2.0))))
                        # overlap if seg midpoint inside interval or any overlap
                        mid = (s+e)/2.0
                        if (mid >= it['start'] and mid <= it['end']) or (s < it['end'] and e > it['start']):
                            t = str(seg.get("text", "")).strip()
                            if t:
                                txts.append(t)
                                segs.append({"start": s, "end": e, "text": t})
                    joined = '\n'.join([t for t in txts if t])
                    it['asr_text'] = joined
                    it['asr_segments'] = segs[:50]
                    # Link sleeping to lesson topics (coarse)
                    topics = []
                    for ch in lesson_timeline:
                        if not isinstance(ch, dict):
                            continue
                        cs = float(ch.get("start", 0.0))
                        ce = float(ch.get("end", cs))
                        if (cs < it["end"]) and (ce > it["start"]):
                            tp = str(ch.get("topic", "")).strip()
                            if tp:
                                topics.append(tp)
                    # keep unique in order
                    seen = set()
                    it["lecture_topics"] = [t for t in topics if not (t in seen or seen.add(t))]
                    # Summarize interval into knowledge points (fine-grained)
                    it['knowledge_points'] = _summarize_text(
                        joined or '',
                        llm=llm_client,
                        allow_llm=allow_llm,
                        language=llm_lang,
                    )

            # Save stats file
            _proc_jobs[job_id]["status"] = "running:write_stats"
            stats_out = session_dir / 'stats.json'
            with open(stats_out, 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': session_dir.name,
                    'video': str(video_path.name) if video_path else None,
                    'audio': 'temp_audio.wav' if (session_dir / 'temp_audio.wav').exists() else None,
                    'transcript': str(transcript_txt.name) if transcript_txt.exists() else None,
                    'model_config': _redact_model_cfg(model_cfg),
                    'languages': {
                        'llm': llm_lang,
                        'asr': asr_lang,
                    },
                    'tracking': id_switch_stats,
                    'warnings': warnings,
                    'lesson_summary': lesson_summary if lesson_summary else None,
                    'per_student': per_student,
                }, fh, ensure_ascii=False, indent=2)

            # mark complete
            _proc_jobs[job_id]['status'] = 'done'
            _proc_jobs[job_id]['result'] = {
                'video': f"/out/{session_dir.name}/{video_path.name}" if video_path else None,
                'audio': f"/out/{session_dir.name}/temp_audio.wav" if (session_dir / 'temp_audio.wav').exists() else None,
                'transcript': f"/out/{session_dir.name}/{transcript_txt.name}" if transcript_txt.exists() else None,
                'stats': f"/out/{session_dir.name}/{stats_out.name}",
                'lesson_summary': f"/out/{session_dir.name}/{lesson_summary_path.name}" if lesson_summary else None,
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
        "session_id": session_mgr.session_id,
        "model_config": _get_model_cfg(),
    }


@app.post("/api/session/rename")
async def rename_session(request: Request):
    """Rename a session by setting a display name (stored in metadata.json)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = body.get("session_id")
    new_name = body.get("name", "").strip()
    if not sid:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    if not new_name:
        return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
    
    session_dir = OUT_DIR / sid
    if not session_dir.exists():
        return JSONResponse({"ok": False, "error": f"session not found: {sid}"}, status_code=404)
    
    # Store display name in metadata.json
    meta_path = session_dir / "metadata.json"
    meta = {}
    try:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
    except Exception:
        meta = {}
    
    meta["display_name"] = new_name
    meta["renamed_at"] = time.time()
    
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Session {sid} renamed to '{new_name}', metadata saved to {meta_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata for {sid}: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    
    return {"ok": True, "session_id": sid, "name": new_name}


# ===== WebDAV API Endpoints =====

@app.get("/api/webdav/config")
async def get_webdav_config():
    """Get WebDAV configuration (password redacted)."""
    cfg = _get_webdav_cfg()
    return {"ok": True, "config": _redact_webdav_cfg(cfg)}


@app.post("/api/webdav/config")
async def set_webdav_config(request: Request):
    """Set WebDAV configuration."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    cfg_data = body.get("config") or body
    
    # If password is "***" (redacted), keep the existing password
    if cfg_data.get("password") == "***":
        existing = _get_webdav_cfg()
        cfg_data["password"] = existing.get("password", "")
    
    cfg = _set_webdav_cfg(cfg_data)
    return {"ok": True, "config": _redact_webdav_cfg(cfg)}


@app.post("/api/webdav/test")
async def test_webdav_connection(request: Request):
    """Test WebDAV connection with provided or saved config."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    # Use provided config or fall back to saved config
    cfg_data = body.get("config") or body
    if not cfg_data or not cfg_data.get("url"):
        cfg_data = _get_webdav_cfg()
    else:
        # If password is redacted, use saved password
        if cfg_data.get("password") == "***":
            existing = _get_webdav_cfg()
            cfg_data["password"] = existing.get("password", "")
    
    try:
        from sync.webdav_client import WebDAVConfig, WebDAVClient
        config = WebDAVConfig.from_dict(cfg_data)
        if not config.is_valid():
            return JSONResponse({"ok": False, "error": "Invalid configuration: URL and username required"}, status_code=400)
        
        client = WebDAVClient(config)
        result = client.test_connection()
        return result
    except ImportError:
        return JSONResponse({"ok": False, "error": "WebDAV client not available. Install webdavclient3."}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/webdav/upload")
async def upload_to_webdav(request: Request, background_tasks: BackgroundTasks):
    """Upload a session to WebDAV."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    session_id = body.get("session_id") or session_mgr.session_id
    if not session_id:
        return JSONResponse({"ok": False, "error": "session_id required"}, status_code=400)
    
    session_dir = OUT_DIR / session_id
    if not session_dir.exists():
        return JSONResponse({"ok": False, "error": f"Session not found: {session_id}"}, status_code=404)
    
    cfg = _get_webdav_cfg()
    if not cfg.get("enabled"):
        return JSONResponse({"ok": False, "error": "WebDAV is not enabled"}, status_code=400)
    
    try:
        from sync.webdav_client import WebDAVConfig, WebDAVClient
        config = WebDAVConfig.from_dict(cfg)
        if not config.is_valid():
            return JSONResponse({"ok": False, "error": "Invalid WebDAV configuration"}, status_code=400)
        
        client = WebDAVClient(config)
        result = client.upload_session(str(session_dir), session_id)
        return result
    except ImportError:
        return JSONResponse({"ok": False, "error": "WebDAV client not available. Install webdavclient3."}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/webdav/sessions")
async def list_webdav_sessions():
    """List sessions available on WebDAV."""
    cfg = _get_webdav_cfg()
    if not cfg.get("enabled"):
        return JSONResponse({"ok": False, "error": "WebDAV is not enabled"}, status_code=400)
    
    try:
        from sync.webdav_client import WebDAVConfig, WebDAVClient
        config = WebDAVConfig.from_dict(cfg)
        if not config.is_valid():
            return JSONResponse({"ok": False, "error": "Invalid WebDAV configuration"}, status_code=400)
        
        client = WebDAVClient(config)
        result = client.list_remote_sessions()
        return result
    except ImportError:
        return JSONResponse({"ok": False, "error": "WebDAV client not available. Install webdavclient3."}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ===== Settings API (combined settings) =====

@app.get("/api/settings")
async def get_all_settings():
    """Get all settings (model config + WebDAV config)."""
    return {
        "ok": True,
        "model": _get_model_cfg(),
        "webdav": _redact_webdav_cfg(_get_webdav_cfg()),
        "env": _env_summary(),
    }


@app.post("/api/settings")
async def set_all_settings(request: Request):
    """Set all settings."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    result = {"ok": True}
    
    if "model" in body:
        cfg = _sanitize_model_cfg(body["model"])
        cfg = _set_model_cfg(cfg)
        result["model"] = cfg
    
    if "webdav" in body:
        webdav_data = body["webdav"]
        if webdav_data.get("password") == "***":
            existing = _get_webdav_cfg()
            webdav_data["password"] = existing.get("password", "")
        cfg = _set_webdav_cfg(webdav_data)
        result["webdav"] = _redact_webdav_cfg(cfg)
    
    result["env"] = _env_summary()
    return result


@app.get("/api/models/config")
async def get_models_config():
    return {
        "ok": True,
        "config": _get_model_cfg(),
        "env": _env_summary(),
        "providers": {
            "llm": ["openai_compat", "none"],
            "asr": ["openai_compat", "dashscope", "xfyun_raasr", "none"],
        },
    }


@app.post("/api/models/config")
async def set_models_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    cfg = _sanitize_model_cfg((body or {}).get("config") or body)
    cfg = _set_model_cfg(cfg)
    return {"ok": True, "config": cfg, "env": _env_summary()}


@app.post("/api/models/list")
async def list_models(request: Request):
    """List model ids from the configured OpenAI-compatible base URL."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    cfg = _sanitize_model_cfg((body or {}).get("config") or _get_model_cfg())
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    provider = str(llm_cfg.get("provider") or "none")
    if provider == "none":
        return JSONResponse({"ok": False, "error": "llm provider is none"}, status_code=400)

    client = _openai_client_from_cfg(llm_cfg)
    if not client:
        return JSONResponse(
            {
                "ok": False,
                "error": "API key not set",
                "hints": _hints_for_openai_compat(llm_cfg, context="llm", err="API key not set"),
            },
            status_code=400,
        )
    try:
        ids = client.list_model_ids()
        return {"ok": True, "models": ids, "count": len(ids)}
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "hints": _hints_for_openai_compat(llm_cfg, context="llm", err=str(exc)),
            },
            status_code=400,
        )


@app.post("/api/models/check")
async def check_models(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    cfg = _sanitize_model_cfg((body or {}).get("config") or _get_model_cfg())
    deep = bool((body or {}).get("deep", False))
    res = _check_models(cfg, deep=deep)
    # Suggest offline if a required provider fails.
    suggested = "online"
    if cfg.get("mode") == "online":
        if (cfg.get("asr") or {}).get("provider") not in ("none", ""):
            if not bool((res.get("asr") or {}).get("ok", False)) and not bool((res.get("asr") or {}).get("skipped", False)):
                suggested = "offline"
        if bool((cfg.get("llm") or {}).get("enabled")) and (cfg.get("llm") or {}).get("provider") != "none":
            if not bool((res.get("llm") or {}).get("ok", False)) and not bool((res.get("llm") or {}).get("skipped", False)):
                suggested = "offline"
    return {"ok": True, "result": res, "suggested_mode": suggested}


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
    dropped = 0
    for e in events:
        if isinstance(e, dict) and "ts" not in e:
            e["ts"] = asyncio.get_event_loop().time()
        if not _enqueue_event(e):
            dropped += 1
        stats['received'] += 1

    return JSONResponse({"ok": True, "queued": len(events) - dropped, "dropped": dropped})


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
