"""Generate a simulated classroom session: WAV audio, ASR JSONL, and face-tracks JSONL.

The ASR JSONL contains simple text segments with timestamps; face_tracks JSONL contains
per-frame / per-ts face observations for a teacher track and some student tracks.
"""
import json
import wave
from pathlib import Path
from typing import List

import numpy as np


def synth_wav(path: str, duration: float, sr: int = 16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # create segments: teacher tone (440), student tone (660), silence patterns
    sig = np.zeros_like(t)
    # teacher segments: 0-5s, 8-13s
    def add_tone(s, e, f, amp=0.2):
        si = int(s * sr)
        ei = int(e * sr)
        sig[si:ei] += amp * np.sin(2 * np.pi * f * t[si:ei])

    add_tone(0.0, 5.0, 440.0)
    add_tone(5.0, 8.0, 660.0)
    add_tone(8.0, 13.0, 440.0)
    add_tone(13.0, 17.0, 660.0)
    # small fade to avoid clicks
    sig = sig * 0.9
    # convert to int16
    pcm = (sig * 32767).astype('int16')
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def gen_asr_jsonl(path: str):
    events = [
        {"start": 0.2, "end": 4.8, "text": "大家好，今天我们讲解线性代数的基础。"},
        {"start": 5.1, "end": 7.5, "text": "老师，我有一个问题。"},
        {"start": 8.0, "end": 12.5, "text": "继续演示矩阵乘法的例子。"},
        {"start": 13.2, "end": 16.8, "text": "学生讨论作业问题。"},
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + '\n')


def gen_face_tracks(path: str):
    # produce per-0.5s observations
    recs = []
    ts = 0.0
    while ts <= 18.0:
        # teacher (T) present mostly centered and large during teacher speech
        if (0.0 <= ts <= 5.0) or (8.0 <= ts <= 13.0):
            # teacher large centered
            recs.append({"ts": round(ts, 3), "track_id": "T", "bbox": [0.25, 0.2, 0.5, 0.5]})
        else:
            # teacher maybe absent or small
            if ts % 7.0 < 1.0:
                recs.append({"ts": round(ts, 3), "track_id": "T", "bbox": [0.4, 0.2, 0.2, 0.25]})
        # student S1 at right
        if ts >= 4.0:
            recs.append({"ts": round(ts, 3), "track_id": "S1", "bbox": [0.8, 0.3, 0.08, 0.1]})
        # student S2 occasionally
        if (6.0 <= ts <= 10.0) and (ts % 3.0 < 1.5):
            recs.append({"ts": round(ts, 3), "track_id": "S2", "bbox": [0.05, 0.3, 0.07, 0.09]})
        ts += 0.5
    with open(path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--duration', type=float, default=18.0)
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_prefix) + '.wav'
    asr_path = str(out_prefix) + '.asr.jsonl'
    faces_path = str(out_prefix) + '.faces.jsonl'

    synth_wav(wav_path, args.duration)
    gen_asr_jsonl(asr_path)
    gen_face_tracks(faces_path)

    print('Generated:', wav_path, asr_path, faces_path)


if __name__ == '__main__':
    main()
