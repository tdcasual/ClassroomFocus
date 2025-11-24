#!/usr/bin/env python3
"""更彻底的摄像头探测脚本：尝试多个后端与更大索引范围，打印检测结果。"""
import json
import time
import cv2
import importlib.util
from pathlib import Path

# 直接按文件路径导入 face_analyzer，避免包导入问题
fa_path = Path(__file__).resolve().parents[1] / "cv" / "face_analyzer.py"
spec = importlib.util.spec_from_file_location("face_analyzer", str(fa_path))
face_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_analyzer)
list_cameras = face_analyzer.list_cameras
open_camera = face_analyzer.open_camera

BACKENDS = [
    ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", None)),
    ("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)),
    ("CAP_ANY", getattr(cv2, "CAP_ANY", None)),
]

results = []
for name, backend in BACKENDS:
    if backend is None:
        print(f"后端 {name} 在此 OpenCV 构建中不可用，跳过")
        continue
    print(f"探测后端 {name} ...")
    cams = list_cameras(max_index=64, backend=backend, timeout=0.6)
    entry = {"backend": name, "found": cams, "probes": []}
    for c in cams:
        idx = c["index"]
        print(f"  尝试打开 index={idx} ...")
        cap = open_camera(idx, width=640, height=480, backend=backend)
        ok = False
        info = {"index": idx}
        if cap.isOpened():
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
        else:
            info.update({"read": False, "error": "open_failed"})
        try:
            cap.release()
        except Exception:
            pass
        entry["probes"].append(info)
    results.append(entry)

print(json.dumps(results, ensure_ascii=False, indent=2))
