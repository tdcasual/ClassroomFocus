"""Replay JSONL events to the /push endpoint with optional timestamp alignment."""
import argparse
import json
import time
from typing import Any, Dict, Iterable, Optional

import requests


def _extract_event_time(event: Dict[str, Any]) -> Optional[float]:
    """Return a representative timestamp for the event if available."""
    for key in ("ts", "start", "time", "timestamp"):
        if key in event:
            try:
                return float(event[key])
            except (TypeError, ValueError):
                continue
    return None


def _iter_events(file_path: str) -> Iterable[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def replay(file_path: str,
           push_url: str,
           speed: float = 1.0,
           interval: float = 0.05,
           align_now: bool = False) -> None:
    events = list(_iter_events(file_path))
    if not events:
        print('no events found')
        return

    base_event_time = None
    base_wall_time = None

    for idx, ev in enumerate(events):
        if align_now:
            ev_time = _extract_event_time(ev)
            if ev_time is not None:
                if base_event_time is None:
                    base_event_time = ev_time
                    base_wall_time = time.time()
                target_wall = base_wall_time + (ev_time - base_event_time) / max(speed, 1e-6)
                delay = target_wall - time.time()
                if delay > 0:
                    time.sleep(delay)
            else:
                # fallback when event has no timestamp metadata
                if idx > 0:
                    time.sleep(max(0.0, interval / max(speed, 1e-6)))
        else:
            if idx > 0:
                time.sleep(max(0.0, interval / max(speed, 1e-6)))

        try:
            r = requests.post(push_url, json=ev, timeout=5)
            print('pushed', r.status_code)
        except Exception as exc:
            print('error pushing', exc)


def main():
    parser = argparse.ArgumentParser(description="Replay JSONL events to the web viz server /push endpoint.")
    parser.add_argument('file', help='JSONL file containing events')
    parser.add_argument('--push-url', default='http://localhost:8000/push', help='HTTP endpoint to POST events to')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed multiplier when aligning by timestamps (default 1.0)')
    parser.add_argument('--interval', type=float, default=0.05, help='Fallback fixed interval between events (seconds)')
    parser.add_argument('--align-now', action='store_true', help='Align playback to wall-clock based on event timestamps')
    args = parser.parse_args()

    replay(args.file, args.push_url, args.speed, args.interval, args.align_now)


if __name__ == '__main__':
    main()
