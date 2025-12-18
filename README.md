# Classroom Focus Analyzer | 教室专注度分析仪

通过摄像头分析学生专注状态（眨眼/低头/疑似睡眠/走神），同时录制整节课音频；课后将音频转写为文本并用 OpenAI 兼容大模型整理课程知识点，再把“不专注时间段（走神/疑似睡觉）”与“讲课内容/知识点”对齐，输出可视化报告。

## Features | 功能
- Multi-face tracking + state machine：EAR（眼睛开合）/Pitch（俯仰）→ BLINK / DROWSY / LOOKING_DOWN 等事件。
- 全程录制：`session.mp4`（视频+音频）+ `faces.jsonl`/`cv_events.jsonl`（结构化事件）+ `sync.json`（时间基准对齐）。
- 报告生成：整节课音频 ASR → `transcript.txt` → LLM 总结/时间线 → 将“不专注区间（闭眼/低头/眼睛不可见）”与讲课主题/ASR 文本关联。
- Web UI：实时预览、事件时间线、历史 Session 列表、一键生成报告。

## Requirements | 环境依赖
- Python 3.12 recommended（MediaPipe 通常对版本更敏感；3.13 可能需要自行处理依赖）。
- Hardware: webcam + microphone。
- System tools: `ffmpeg`（用于音视频 mux）。
- Python deps：见 `requirements.txt`（包含 `opencv-python` / `mediapipe` / `sounddevice` / `soundfile` / `fastapi` / `uvicorn` 等）。
- DashScope 实时流式 ASR 脚本是可选功能（需要 `dashscope` + `pyaudio`；如安装 `pyaudio` 困难，可先只用 Web 录制+离线 ASR 报告链路）。

## Quick Start | 快速上手
```bash
python3 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

pip install -r requirements.txt
# Optional:
#   pip install -r requirements-dev.txt        # pytest
#   pip install -r requirements-optional.txt   # dashscope/pyaudio/matplotlib
```

### Config (.env) | 配置
创建 `.env`（按需配置）：
- DashScope（可选实时 ASR）：`DASH_SCOPE_API_KEY=...`（或 `DASHSCOPE_API_KEY`）。
- OpenAI-compatible（用于“整节课离线 ASR + 课程总结/知识点整理”）：
  - `OPENAI_BASE_URL=https://api.openai.com`
  - `OPENAI_API_KEY=...`
  - `OPENAI_MODEL=...`（支持 GPT-5+ 或旧式 Chat Completions 的兼容后端）
  - `OPENAI_ASR_MODEL=whisper-1`
- Optional:
  - `ASR_CHUNK_SECONDS=60`（长音频分块转写）
  - `SUMMARY_CHUNK_SECONDS=180`（按时间块做课程时间线）
  - `FFMPEG_PATH=ffmpeg`
  - `WEB_PREVIEW_INTERVAL_SEC=0.15`（Web 预览帧率节流）
  - `NOT_VISIBLE_GAP_SEC=1.6`（人脸轨迹中断多久算“眼睛不可见/趴下”）
  - `NOT_VISIBLE_TAIL_CAP_SEC=12`（末尾“不可见”区间最长补齐秒数；<=0 表示不限制）
  - `INATTENTIVE_MERGE_GAP_SEC=0.4`（多种不专注区间合并的最大间隙）

> 兼容性说明：`tools/openai_compat.py` 会优先尝试 `/v1/responses`（常见于 GPT-5+ 风格模型），失败后回退到 `/v1/chat/completions`；并自动兼容 `max_output_tokens`/`max_tokens`/`max_completion_tokens` 参数差异。

## Run (Recommended: Web UI) | 运行（推荐：Web UI）
```bash
./.venv/bin/python tools/web_viz_server.py
# open http://localhost:8000/
```

Web UI 交互流程：
1) Start Session（开始录制）
2) Stop Session（停止并落盘）
3) Generate Report（离线 ASR + LLM 总结 + 不专注区间关联讲课内容）

## Outputs | 输出文件
每次录制会生成一个目录：`out/<session_id>/`
- 录制：`temp_video.avi`、`temp_audio.wav`、`session.mp4`
- CV：`faces.jsonl`、`cv_events.jsonl`
- 同步：`sync.json`（用于把音频/ASR 时间与 CV 时间对齐）
- 报告：`asr.jsonl`、`transcript.txt`、`lesson_summary.json`、`stats.json`

`stats.json` 中会包含每个 track 的不专注区间（走神/疑似睡觉）、区间内的 ASR 文本片段，以及与课程主题（`lesson_summary.timeline`）的关联结果。

## Camera Motion | 摄像头平移抖动/旋转变焦
当前实现采用“时间感知匹配 + 运动预测 + 全局相机位移补偿 +（可选）相似变换补偿”的组合策略来尽量保证同一节课 track 不乱（见 `cv/face_analyzer.py`）。

如遇到“大幅变焦/旋转 + 多个相似人脸”的极端情况，建议引入轻量人脸特征（embedding）做 ReID 约束：仅在几何匹配不确定时做一次 embedding 比对，以降低成本并提升鲁棒性。

## Other Scripts | 其他脚本
- Integrated CV + ASR demo: `./.venv/bin/python tools/integrated_cv_asr_test.py --duration 20 --show --webcam 0`
- Video replay: `./.venv/bin/python tools/run_video_replay.py`（示例视频见 `samples/`）
- Webcam probing: `./.venv/bin/python tools/list_cams_probe.py`、`./.venv/bin/python tools/try_open_by_name.py`

## Tests | 测试
```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/pytest
```

## Project Layout | 项目结构
- `cv/face_analyzer.py` – FaceMesh 跟踪与状态机（含相机运动补偿与 track 匹配）。
- `tools/session_manager.py` – 录制整节课音视频+CV 事件，输出 `out/<session>/...`。
- `tools/web_viz_server.py` – FastAPI Web UI + 报告生成链路（ASR + LLM 总结 + 不专注区间关联）。
- `tools/openai_compat.py` – OpenAI 兼容调用封装（Responses/ChatCompletions/ASR）。
- `analysis/teacher_labeler.py` – 基于 face track 的 teacher/student 粗分标注工具。
- `tests/` – 纯 Python 单测（无需摄像头/麦克风/网络）。

## Troubleshooting | 排障
- 摄像头打不开：先试 `tools/list_cams_probe.py`、`tools/try_open_by_name.py`，或在 Windows 切换后端（如 `cv2.CAP_DSHOW`）。
- 麦克风无输入（Web 录制链路）：确认系统麦克风权限；Linux 可能需要 `portaudio` 相关依赖。
- 报告生成失败：检查 `.env` 的 `OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL` 与网络；ASR/LLM 会产生调用成本。
