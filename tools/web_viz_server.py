import asyncio
import json
import base64
import os
import time
import re
import tempfile
import shutil
from typing import Any, Dict, List, Optional
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
from analysis.inattentive_intervals import infer_not_visible_intervals, merge_inattentive_intervals

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

# in-memory queue of events
event_queue: asyncio.Queue = asyncio.Queue()

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
}

# ---- Model config (in-memory defaults for next session) ----
_MODEL_CFG: Dict[str, Any] = {}


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
    return {
        "mode": "online" if has_openai else "offline",
        "llm": {
            "enabled": bool(has_openai),
            "provider": "openai_compat",
            "model": str(env.get("openai_model") or "gpt-4o-mini"),
            "base_url": str(env.get("openai_base_url") or "https://api.openai.com"),
            # Optional override (kept in memory only; never persisted to session outputs).
            "api_key": "",
        },
        "asr": {
            "provider": "openai_compat" if has_openai else "none",
            "model": str(env.get("openai_asr_model") or "whisper-1"),
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

    asr = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
    asr_provider = str(asr.get("provider", out["asr"]["provider"]) or "").strip()
    if asr_provider not in ("openai_compat", "dashscope", "xfyun_raasr", "none"):
        asr_provider = out["asr"]["provider"]
    out["asr"]["provider"] = asr_provider
    asr_model = str(asr.get("model", out["asr"]["model"]) or "").strip()
    if asr_model:
        out["asr"]["model"] = asr_model[:120]
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
    return out


def _write_session_config(session_dir: Path, cfg: Dict[str, Any]) -> None:
    try:
        with open(session_dir / "session_config.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_config": _redact_model_cfg(cfg),
                    "env_summary": _env_summary(),
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
                if isinstance(loaded.get("llm"), dict) and isinstance(cur.get("llm"), dict):
                    if not str(loaded["llm"].get("api_key") or "").strip() and str(cur["llm"].get("api_key") or "").strip():
                        loaded["llm"]["api_key"] = cur["llm"]["api_key"]
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
            client = _openai_client_from_cfg(cfg.get("llm") or {})
            if not client:
                out["asr"] = {
                    "ok": False,
                    "error": "API key not set",
                    "hints": _hints_for_openai_compat(cfg.get("llm") or {}, context="asr", err="API key not set"),
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
                        "hints": _hints_for_openai_compat(cfg.get("llm") or {}, context="asr", err=str(exc)),
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
    base = OUT_DIR
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
async def start_session(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        cfg = _sanitize_model_cfg((body or {}).get("model_config") or (body or {}).get("config") or _get_model_cfg())
        # Persist as the current default so the UI/session status stays consistent.
        cfg = _set_model_cfg(cfg)

        sid = session_mgr.start(output_dir_base=str(OUT_DIR))
        # Persist config snapshot to session dir for report reproducibility.
        try:
            session_dir = OUT_DIR / sid
            _write_session_config(session_dir, cfg)
        except Exception:
            pass
        return {"ok": True, "session_id": sid, "model_config": cfg}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/session/stop")
async def stop_session():
    try:
        path = session_mgr.stop()
        warn = getattr(session_mgr, "video_error", None) or None
        return {"ok": True, "path": path, "warning": warn}
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


def _summarize_text(text: str, max_points: int = 6, llm: Optional[OpenAICompat] = None, allow_llm: bool = True):
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
            messages = [
                {"role": "system", "content": "你从课堂讲解中提炼知识点。只输出 JSON 数组（字符串数组），不要输出任何额外文字。"},
                {"role": "user", "content": f"从下面转录内容中提炼最多 {int(max_points)} 条知识点（尽量中文、短语化、去重）：\n\n{text}"},
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


def _llm_summarize_lesson(asr_segments: List[dict], llm: Optional[OpenAICompat] = None, allow_llm: bool = True) -> dict:
    """Summarize the whole lesson and produce a timeline of topics."""
    if not allow_llm:
        return {}
    if llm is None:
        llm = OpenAICompat.from_env()
    if not llm:
        return {}
    chunk_secs = float(os.getenv("SUMMARY_CHUNK_SECONDS", "180"))
    chunks = _chunk_asr_segments(asr_segments, chunk_secs=chunk_secs)
    if not chunks:
        return {}

    timeline = []
    for ch in chunks:
        messages = [
            {"role": "system", "content": "你要总结课堂讲解内容。只输出 JSON 对象，不要输出任何额外文字。"},
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
        {"role": "system", "content": "你要生成用于课堂专注度分析的课程结构化总结。只输出 JSON 对象，不要输出任何额外文字。"},
        {
            "role": "user",
            "content": (
                "根据下面按时间分段的总结，生成整节课的总结。\n"
                "只返回 JSON，字段为：\n"
                "- title：课程标题\n"
                "- overview：3-6 句概览\n"
                "- key_points：6-12 条关键要点（短语化）\n"
                "- outline：大纲（章节标题数组）\n"
                "要求中文。\n\n"
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


def _write_asr_segments(asr_path: Path, segments: List[dict]) -> None:
    try:
        with open(asr_path, "w", encoding="utf-8") as fh:
            for s in segments:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _transcribe_openai_audio_to_segments(session_dir: Path, job_id: str, client: OpenAICompat, asr_model: Optional[str]) -> List[dict]:
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
                    resp = client.transcribe_audio(str(chunk_path), model=asr_model or None, response_format="verbose_json")
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
    provider = str(asr_cfg.get("provider") or "none")
    model = str(asr_cfg.get("model") or "").strip() or None

    try:
        if provider == "none":
            return []
        if provider == "openai_compat":
            client = _openai_client_from_cfg(model_cfg.get("llm") or {})
            if not client:
                raise RuntimeError("OPENAI_API_KEY not set.")
            segs = _transcribe_openai_audio_to_segments(session_dir, job_id, client, model)
        elif provider == "dashscope":
            segs = _transcribe_dashscope_audio_to_segments(session_dir, job_id, model)
        elif provider == "xfyun_raasr":
            segs = _transcribe_xfyun_audio_to_segments(session_dir, job_id)
        else:
            raise RuntimeError(f"unknown ASR provider: {provider}")
    except Exception as exc:
        warnings.append(f"ASR failed ({provider}): {exc}")
        return []

    _write_asr_segments(asr_path, segs)
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
            lesson_summary = _llm_summarize_lesson(asr_segments, llm=llm_client, allow_llm=allow_llm)
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
                    it['knowledge_points'] = _summarize_text(joined or '', llm=llm_client, allow_llm=allow_llm)

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
