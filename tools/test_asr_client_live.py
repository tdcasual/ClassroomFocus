"""Live ASR client test using microphone + DashScope.

Usage:
  - Set your API key in env var `DASHSCOPE_API_KEY` or pass as first arg.
  - Run for N seconds (default 20) then exit.

This script requires `dashscope` to be installed and a valid API key.
"""
import os
import time
import sys
from typing import Dict
from pathlib import Path

# Ensure project root is on sys.path so `import asr.asr_client` works
proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from asr.asr_client import AliASRClient


def time_base():
    return time.time()


def on_sentence(ev: Dict):
    # 尽量提取纯文本输出：
    txt = None
    t = ev.get('text') if isinstance(ev, dict) else None
    if isinstance(t, str):
        txt = t
    elif isinstance(t, dict):
        # 首选直接的 'text' 字段
        txt = t.get('text')
        if not txt:
            # 退回到 words 列表拼接
            words = t.get('words') or []
            txt = ''.join([w.get('text', '') for w in words]) if words else ''
    # 最终回退：把整个 ev 转为字符串
    if not txt:
        txt = str(ev)

    print(txt)


if __name__ == '__main__':
    api_key = None
    dur = 20
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get('DASHSCOPE_API_KEY')
    if len(sys.argv) > 2:
        try:
            dur = int(sys.argv[2])
        except Exception:
            pass

    print('API key provided:' , bool(api_key))
    client = AliASRClient(api_key=api_key, time_base=time_base, on_sentence=on_sentence)
    print('Starting ASR client (speak into microphone)...')
    client.start()
    try:
        time.sleep(dur)
    except KeyboardInterrupt:
        pass
    print('Stopping ASR client...')
    client.stop()
    print('Done')
