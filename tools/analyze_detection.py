"""Analyze face EAR/pitch traces vs CV-detected events to find missed drowsy periods.

Usage:
  python tools/analyze_detection.py --faces out/demo_demo.faces.jsonl --events out/demo_demo.cv_events.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def find_low_ear_spans(frames: List[Dict], ear_thresh: float = 0.18, min_dur: float = 2.8):
    # frames: list of records with ts, track_id, ear
    by_track = defaultdict(list)
    for r in frames:
        if 'ear' in r and r['ear'] is not None:
            by_track[str(r.get('track_id'))].append((r['ts'], float(r['ear'])))

    spans = []
    for tid, samples in by_track.items():
        samples.sort()
        cur_start = None
        last_t = None
        for t, ear in samples:
            if ear < ear_thresh:
                if cur_start is None:
                    cur_start = t
                last_t = t
            else:
                if cur_start is not None:
                    dur = last_t - cur_start
                    if dur >= min_dur:
                        spans.append({'track_id': tid, 'start': cur_start, 'end': last_t, 'dur': dur})
                    cur_start = None
                    last_t = None
        if cur_start is not None:
            dur = last_t - cur_start
            if dur >= min_dur:
                spans.append({'track_id': tid, 'start': cur_start, 'end': last_t, 'dur': dur})
    return spans


def find_events(events: List[Dict], ev_type: str = 'DROWSY_START'):
    return [e for e in events if e.get('type') == ev_type]


def overlaps(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or b_end < a_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--ear-thresh', type=float, default=0.18)
    ap.add_argument('--drowsy-secs', type=float, default=2.8)
    args = ap.parse_args()

    faces = load_jsonl(Path(args.faces))
    events = load_jsonl(Path(args.events))

    spans = find_low_ear_spans(faces, ear_thresh=args.ear_thresh, min_dur=args.drowsy_secs)
    drowsy_starts = find_events(events, 'DROWSY_START')
    drowsy_periods = []
    for ds in drowsy_starts:
        tid = str(ds.get('student_id'))
        st = ds.get('ts')
        # find corresponding DROWSY_END
        end = None
        for e in events:
            if e.get('type') == 'DROWSY_END' and e.get('student_id') == ds.get('student_id') and e.get('ts') >= st:
                end = e.get('ts')
                break
        if end is None:
            end = st + args.drowsy_secs
        drowsy_periods.append({'track_id': tid, 'start': st, 'end': end})

    missed = []
    for s in spans:
        tid = s['track_id']
        matched = False
        for p in drowsy_periods:
            if p['track_id'] == tid and overlaps(s['start'], s['end'], p['start'], p['end']):
                matched = True
                break
        if not matched:
            missed.append(s)

    print(f"Total low-ear spans (ear<{args.ear_thresh}, dur>={args.drowsy_secs}): {len(spans)}")
    print(f"Total DROWSY_START events: {len(drowsy_starts)}")
    print(f"Missed spans (no overlapping DROWSY_START): {len(missed)}")
    if missed:
        print("Examples:")
        for m in missed[:10]:
            print(json.dumps(m, ensure_ascii=False))


if __name__ == '__main__':
    main()
