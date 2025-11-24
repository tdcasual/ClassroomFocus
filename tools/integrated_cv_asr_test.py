"""Integrated CV + ASR short test.

Runs FaceAnalyzer on the default webcam and AliASRClient on the microphone
concurrently, printing merged events to the console.

Usage:
  .venv\\Scripts\\python.exe tools\\integrated_cv_asr_test.py [--duration SECS] [--show]

Controls:
  - Press ESC in the OpenCV window to exit early when `--show` is used.
"""
import argparse
import threading
import time
from pathlib import Path
from typing import List, Dict

proj_root = Path(__file__).resolve().parents[1]
import sys
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import cv2
from dotenv import load_dotenv
import os
from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig, open_camera
from asr.asr_client import AliASRClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--webcam', type=int, default=0)
    args = ap.parse_args()

    shared = {
        'asr_events': [],
        'face_snapshot': {},
        'lock': threading.Lock(),
    }

    def time_base():
        return time.time()

    def on_sentence(ev: Dict):
        # Called from ASR thread
        with shared['lock']:
            shared['asr_events'].append(ev)
            # print merged snapshot: latest face states + asr text
            faces = list(shared['face_snapshot'].values())
            text = ''
            t = ev.get('text')
            if isinstance(t, dict):
                text = t.get('text') or ''.join([w.get('text','') for w in (t.get('words') or [])])
            elif isinstance(t, str):
                text = t
            print(f"[ASR @ {ev['ts']:.3f}] {text}  | faces={len(faces)}")

    # load .env and map DASH_SCOPE_API_KEY -> DASHSCOPE_API_KEY if needed
    load_dotenv(proj_root / '.env')
    if os.getenv('DASH_SCOPE_API_KEY') and not os.getenv('DASHSCOPE_API_KEY'):
        os.environ['DASHSCOPE_API_KEY'] = os.environ['DASH_SCOPE_API_KEY']

    # start ASR client
    client = AliASRClient(api_key=None, time_base=time_base, on_sentence=on_sentence)
    # If API key present in .env it will be used by AliASRClient.start when called
    client.start()

    # start camera and analyzer
    cfg = FaceAnalyzerConfig(debug_draw=False)
    analyzer = FaceAnalyzer(cfg)
    cap = open_camera(index=args.webcam, width=640, height=480)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.webcam}")
        client.stop()
        return

    t0 = time.time()
    end_t = t0 + args.duration
    frames = 0
    try:
        while time.time() < end_t:
            ok, frame = cap.read()
            if not ok:
                print('Camera read failed, stopping')
                break
            ts = time.time()
            results, events = analyzer.analyze_frame(frame, ts)
            # update snapshot
            with shared['lock']:
                # snapshot: map student_id -> latest result
                for r in results:
                    shared['face_snapshot'][r['student_id']] = r
            # print face events
            for e in events:
                with shared['lock']:
                    faces = list(shared['face_snapshot'].values())
                    print(f"[FACE @ {e.get('ts', ts):.3f}] {e.get('type')} student={e.get('student_id')} faces={len(faces)}")

            frames += 1
            if args.show:
                # draw debug overlay if available
                analyzer.cfg.debug_draw = True
                cv2.imshow('IntegratedTest', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print('ESC pressed, exiting')
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.stop()
        dur = max(1e-6, time.time() - t0)
        print(f"Finished: frames={frames}, FPS={frames/dur:.2f}")


if __name__ == '__main__':
    main()
