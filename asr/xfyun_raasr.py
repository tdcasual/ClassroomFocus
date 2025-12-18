from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class XfyunRaasrConfig:
    app_id: str
    secret_key: str
    host: str = "https://raasr.xfyun.cn/v2/api"
    slice_size_bytes: int = 10 * 1024 * 1024
    poll_interval_sec: float = 2.0
    timeout_sec: float = 300.0


def _signa(app_id: str, ts: str, secret_key: str) -> str:
    raw = (app_id + ts + secret_key).encode("utf-8")
    md5 = hashlib.md5(raw).digest()
    return base64.b64encode(md5).decode("utf-8")


def _slice_id(index: int) -> str:
    """RaaSR slice id: 8 lowercase letters (base-26)."""
    if index < 0:
        index = 0
    chars = []
    x = int(index)
    for _ in range(8):
        chars.append(chr(ord("a") + (x % 26)))
        x //= 26
    return "".join(reversed(chars))


def _post_form(url: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    r = requests.post(url, data=data, files=files, timeout=timeout)
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"xfyun response is not json (HTTP {r.status_code}): {r.text[:2000]}")
    if r.status_code < 200 or r.status_code >= 300:
        raise RuntimeError(f"xfyun HTTP {r.status_code}: {j}")
    return j


def _require_ok(resp: Dict[str, Any], ctx: str) -> None:
    ok = resp.get("ok")
    if ok is None:
        # some variants use err_no
        err_no = resp.get("err_no")
        if err_no in (0, "0", None):
            return
        raise RuntimeError(f"{ctx} failed: {resp}")
    if int(ok) != 0:
        raise RuntimeError(f"{ctx} failed: {resp}")


def _parse_result_data(data_list: List[Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for item in data_list:
        # data items are usually JSON strings
        if isinstance(item, str):
            try:
                obj = json.loads(item)
            except Exception:
                continue
        elif isinstance(item, dict):
            obj = item
        else:
            continue

        text = str(obj.get("onebest") or obj.get("text") or "").strip()
        if not text:
            continue
        bg = obj.get("bg") or obj.get("start") or obj.get("begin") or obj.get("start_time")
        ed = obj.get("ed") or obj.get("end") or obj.get("stop") or obj.get("end_time")
        try:
            s_ms = float(bg) if bg is not None else None
            e_ms = float(ed) if ed is not None else None
        except Exception:
            s_ms, e_ms = None, None
        if s_ms is None:
            s_ms = 0.0
        if e_ms is None or e_ms <= s_ms:
            e_ms = s_ms + 1000.0
        segments.append(
            {
                "start": float(s_ms / 1000.0),
                "end": float(e_ms / 1000.0),
                "text": text,
                "raw": obj,
            }
        )
    segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return segments


def transcribe_wav(
    wav_path: str,
    cfg: XfyunRaasrConfig,
    *,
    file_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Transcribe a WAV file via Xfyun RaaSR (录音文件识别).

    Returns: (segments, meta)
      - segments: [{start,end,text,raw}, ...] in seconds relative to audio start
      - meta: provider-specific metadata (task_id, raw responses)
    """
    p = Path(wav_path)
    if not p.exists() or p.stat().st_size <= 0:
        raise RuntimeError(f"audio file missing/empty: {wav_path}")

    host = (cfg.host or "").rstrip("/")
    file_len = int(p.stat().st_size)
    file_name = file_name or p.name
    slice_size = max(1, int(cfg.slice_size_bytes))
    slice_num = int((file_len + slice_size - 1) // slice_size)

    ts = str(int(time.time()))
    signa = _signa(cfg.app_id, ts, cfg.secret_key)
    common = {"app_id": cfg.app_id, "ts": ts, "signa": signa}

    # 1) prepare
    prepare_url = f"{host}/prepare"
    prepare_data = dict(common)
    prepare_data.update({"file_len": file_len, "file_name": file_name, "slice_num": slice_num})
    prepare_resp = _post_form(prepare_url, prepare_data, timeout=30.0)
    _require_ok(prepare_resp, "prepare")
    task_id = prepare_resp.get("data") or prepare_resp.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"prepare did not return task_id: {prepare_resp}")

    # 2) upload slices
    upload_url = f"{host}/upload"
    upload_resps = []
    with open(p, "rb") as fh:
        for idx in range(slice_num):
            content = fh.read(slice_size)
            if not content:
                break
            ts = str(int(time.time()))
            signa = _signa(cfg.app_id, ts, cfg.secret_key)
            data = {"app_id": cfg.app_id, "ts": ts, "signa": signa, "task_id": task_id, "slice_id": _slice_id(idx)}
            resp = _post_form(upload_url, data, files={"content": ("slice", content)}, timeout=60.0)
            _require_ok(resp, f"upload slice {idx}")
            upload_resps.append(resp)

    # 3) merge
    merge_url = f"{host}/merge"
    ts = str(int(time.time()))
    signa = _signa(cfg.app_id, ts, cfg.secret_key)
    merge_data = {"app_id": cfg.app_id, "ts": ts, "signa": signa, "task_id": task_id, "file_name": file_name}
    merge_resp = _post_form(merge_url, merge_data, timeout=30.0)
    _require_ok(merge_resp, "merge")

    # 4) poll result
    get_url = f"{host}/getResult"
    t0 = time.time()
    last_resp: Optional[Dict[str, Any]] = None
    while True:
        if (time.time() - t0) > float(cfg.timeout_sec):
            raise RuntimeError(f"getResult timeout after {cfg.timeout_sec}s (task_id={task_id})")
        ts = str(int(time.time()))
        signa = _signa(cfg.app_id, ts, cfg.secret_key)
        data = {"app_id": cfg.app_id, "ts": ts, "signa": signa, "task_id": task_id}
        resp = _post_form(get_url, data, timeout=30.0)
        _require_ok(resp, "getResult")
        last_resp = resp

        status = resp.get("status")
        data_field = resp.get("data")
        # "done" cases
        if isinstance(data_field, list):
            segments = _parse_result_data(data_field)
            return segments, {
                "task_id": task_id,
                "prepare": prepare_resp,
                "merge": merge_resp,
                "getResult": last_resp,
                "slice_num": slice_num,
            }
        try:
            if status is not None and int(status) == 3:
                # Some variants return status==3 but keep data in another key.
                # Treat as done with empty data.
                return [], {
                    "task_id": task_id,
                    "prepare": prepare_resp,
                    "merge": merge_resp,
                    "getResult": last_resp,
                    "slice_num": slice_num,
                }
            if status is not None and int(status) == 4:
                raise RuntimeError(f"getResult error status=4: {resp}")
        except Exception:
            pass

        time.sleep(max(0.5, float(cfg.poll_interval_sec)))


def config_from_env() -> Optional[XfyunRaasrConfig]:
    app_id = os.getenv("XFYUN_APP_ID") or os.getenv("IFLYTEK_APP_ID") or ""
    secret = os.getenv("XFYUN_SECRET_KEY") or os.getenv("IFLYTEK_SECRET_KEY") or ""
    if not app_id or not secret:
        return None
    host = os.getenv("XFYUN_RAASR_HOST") or "https://raasr.xfyun.cn/v2/api"
    try:
        slice_size = int(float(os.getenv("XFYUN_RAASR_SLICE_BYTES", str(10 * 1024 * 1024))))
    except Exception:
        slice_size = 10 * 1024 * 1024
    try:
        poll = float(os.getenv("XFYUN_RAASR_POLL_SEC", "2.0"))
    except Exception:
        poll = 2.0
    try:
        timeout = float(os.getenv("XFYUN_RAASR_TIMEOUT_SEC", "300"))
    except Exception:
        timeout = 300.0
    return XfyunRaasrConfig(app_id=app_id, secret_key=secret, host=host, slice_size_bytes=slice_size, poll_interval_sec=poll, timeout_sec=timeout)
