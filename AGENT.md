# Agent Guide: `classroom_focus`

> Note: This repo also provides `AGENTS.md` (same content) so automated agents can load instructions by convention.

## Primary Goal
- Keep a single class session’s `track_id` stable enough for reporting: “谁在什么时间段睡着/低头”能与“当时讲课内容/知识点”对齐。

## Local Setup (Required)
- Use a venv (don’t install into system Python): `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

## Runbook (Recommended Path)
- Start Web UI: `./.venv/bin/python tools/web_viz_server.py` → open `http://localhost:8000/`
- UI flow: Start Session → Stop Session → Generate Report
- Outputs are written to `out/<session_id>/`:
  - Recording: `session.mp4`, `temp_audio.wav`, `temp_video.avi`
  - CV: `faces.jsonl`, `cv_events.jsonl`
  - Sync: `sync.json` (time-base alignment)
  - Report: `asr.jsonl`, `transcript.txt`, `lesson_summary.json`, `stats.json`

## OpenAI-Compatible Models (ASR + LLM)
- Always call via `tools/openai_compat.py` (compat handles GPT‑5+/pre‑5 differences):
  - Prefer `/v1/responses`, fallback to `/v1/chat/completions`
  - Token param fallback: `max_output_tokens` → `max_tokens` → `max_completion_tokens`
- Required env vars for report: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- Optional: `OPENAI_ASR_MODEL` (default `whisper-1`), `ASR_CHUNK_SECONDS`, `SUMMARY_CHUNK_SECONDS`

## Tracking Robustness (Camera Motion)
- Current strategy: time-aware matching + constant-velocity prediction + global camera shift compensation (+ optional similarity compensation for rotation/zoom).
- If the class has strong pan/zoom/rotation and many similar faces, add a *lightweight* face embedding ReID:
  - Only compute embeddings when geometric matching is ambiguous (cost control).
  - Do not persist embeddings unless explicitly needed; prefer in-memory only (privacy).
  - Use embedding as a constraint, not a replacement, to avoid ID switches after occlusions.

## Testing Expectations
- Tests must run without camera/mic/network by default.
- Run: `./.venv/bin/pytest`
- When changing CV/ASR/LLM logic, add/extend unit tests for pure-python parts (e.g. parsing, alignment, OpenAI compat fallbacks).
- Avoid importing heavy optional deps at module import time in code that is used by tests (e.g. `cv2`, `mediapipe`, `pyaudio`).
