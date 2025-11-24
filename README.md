# Classroom Focus Analyzer | 教室专注度分析

## Overview | 概述
- Classroom attentiveness demo: webcam FaceMesh tracking (EAR/pitch) + DashScope streaming ASR for live drowsiness/downward gaze and speech alignment.
- 工程演示：摄像头 FaceMesh（EAR/头部俯仰）+ DashScope 实时语音识别，对课堂瞌睡/低头与发言进行同步。

## Features | 功能
- Eye aspect ratio + pitch-based state machine：BLINK, DROWSY_START/END, LOOKING_DOWN_START/END。
- CV/ASR events aligned on a shared time base via `DataSynchronizer`。
- 视频/音频/事件录制工具，摄像头探测脚本，离线视频回放。
- 可配置阈值、调试绘制，自动读取 `.env` 中的 DashScope API key。

## Requirements | 环境依赖
- Python 3.12 recommended；`pip install -r requirements.txt`。
- Hardware: webcam (CV) + microphone (ASR)。
- `.env` 中放 `DASH_SCOPE_API_KEY=...`（或 `DASHSCOPE_API_KEY`）。
- Windows 如安装 PyAudio 失败，可 `pip install pipwin && pipwin install pyaudio`。

## Quick Start | 快速上手
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
# create .env
echo DASH_SCOPE_API_KEY=your_key_here > .env
```

## How to Run | 如何运行
- Live ASR (mic): `python tools/test_asr_client_live.py [api_key] [duration_secs]`（API key 可来自 `.env`）。
- Fake mic + optional fake dashscope: `python tools/run_live_with_fake_pyaudio.py`（无真实麦克风时验证连通性）。
- Mock ASR (no network/key): `python tools/test_asr_client_mock.py`（注入假 dashscope+pyaudio，输出示例句子）。
- Integrated CV + ASR: `python tools/integrated_cv_asr_test.py --duration 20 --show --webcam 0`。
- Video replay with overlay: `python tools/run_video_replay.py`（使用 `samples/students.mp4`）。
- Webcam probing: `python tools/list_cams_probe.py`，`python tools/try_open_by_name.py`。
- Record classroom ASR + raw audio: `python tools/record_classroom_asr.py [api_key] [duration_secs]`。

## Tests | 测试
```bash
pytest
```
(EAR 几何与 pitch solvePnP smoke tests。)

## Project Layout | 项目结构
- `cv/face_analyzer.py` – FaceMesh 跟踪，EAR/俯仰状态机，调试绘制。
- `asr/asr_client.py` – DashScope 流式 ASR + PyAudio 采集。
- `sync/synchronizer.py` – CV/ASR 事件对齐。
- `replay/recorder.py` – 视频/音频/JSONL 录制。
- `tools/` – 演示、探测、mock、回放、录制脚本。
- `tests/` – EAR 公式与俯仰计算单测。

## Troubleshooting | 排障
- 摄像头打不开：试 `tools/list_cams_probe.py`、`tools/try_open_by_name.py`，或切换后端（如 Windows 用 `cv2.CAP_DSHOW`）。
- ASR 没声音：检查麦克风权限，先用 `run_live_with_fake_pyaudio.py` 排查 API/网络。
- API 报错：确认 `.env` 中 `DASH_SCOPE_API_KEY/DASHSCOPE_API_KEY`，并确保网络可达。
