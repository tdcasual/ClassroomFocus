#!/usr/bin/env python3
"""在 Windows 上获取 PnP 设备名并使用 DirectShow 名称尝试打开摄像头。

输出 JSON，包含尝试的 name 与是否能打开/读帧的结果。
"""
import subprocess
import json
import time
import cv2
from pathlib import Path
import importlib.util

# 导入 face_analyzer.open_camera，按路径导入以避免包问题
fa_path = Path(__file__).resolve().parents[1] / "cv" / "face_analyzer.py"
spec = importlib.util.spec_from_file_location("face_analyzer", str(fa_path))
face_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_analyzer)
open_camera = face_analyzer.open_camera

# 通过 PowerShell 列出可能的摄像头设备
ps_cmd = "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'Camera|Webcam|Imaging|USB 视频|USB Camera' } | Select-Object -ExpandProperty Name"
try:
    out = subprocess.check_output(["powershell", "-Command", ps_cmd], stderr=subprocess.DEVNULL, text=True)
    names = [l.strip() for l in out.splitlines() if l.strip()]
except Exception:
    names = []

results = []
for name in names:
    entry = {"name": name, "tried_sources": []}
    # DirectShow 支持用 'video={device name}' 打开
    ds_source = f"video={name}"
    try:
        cap = open_camera(ds_source, width=640, height=480, backend=cv2.CAP_DSHOW)
        info = {"source": ds_source, "opened": bool(cap.isOpened())}
        if cap.isOpened():
            ok = False
            t0 = time.time()
            while time.time() - t0 < 1.0:
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    info.update({"read": True, "width": w, "height": h})
                    ok = True
                    break
            if not ok:
                info.update({"read": False})
        entry["tried_sources"].append(info)
    except Exception as e:
        entry["tried_sources"].append({"source": ds_source, "error": str(e)})
    finally:
        try:
            cap.release()
        except Exception:
            pass
    results.append(entry)

print(json.dumps({"device_names": names, "results": results}, ensure_ascii=False, indent=2))
