"""Clean ASR JSONL by collapsing partial/duplicate hypotheses per time window.

Usage:
  python tools/clean_asr_jsonl.py --in out/demo_e2e.asr.jsonl --faces out/demo_e2e.faces.jsonl --out-prefix out/demo_e2e.cleaned

Behavior:
  - Groups ASR entries by exact (start,end) and keeps the longest non-empty text for each group.
  - Drops empty-text entries when a non-empty exists for same window.
  - Writes cleaned ASR JSONL and a labeled JSONL using existing teacher_labeler heuristics.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from analysis.teacher_labeler import load_face_tracks_jsonl, label_asr_segments


def load_jsonl(path: Path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_jsonl(path: Path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def clean_asr(events):
    # group by (start,end) exact match
    groups = defaultdict(list)
    for ev in events:
        key = (float(ev.get('start', 0.0)), float(ev.get('end', 0.0)))
        groups[key].append(ev)

    cleaned = []
    for (s, e), items in sorted(groups.items()):
        # choose longest non-empty text; fallback to empty if none
        best = ''
        for it in items:
            txt = (it.get('text') or '').strip()
            if len(txt) > len(best):
                best = txt
        cleaned.append({'start': float(s), 'end': float(e), 'text': best, 'raw': items[-1].get('raw', {})})
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='infile', required=True)
    ap.add_argument('--faces', dest='faces', required=False)
    ap.add_argument('--out-prefix', dest='out_prefix', required=True)
    args = ap.parse_args()

    infile = Path(args.infile)
    faces = Path(args.faces) if args.faces else None
    out_prefix = Path(args.out_prefix)

    events = load_jsonl(infile)
    cleaned = clean_asr(events)
    cleaned_path = out_prefix.with_suffix('.asr.cleaned.jsonl')
    write_jsonl(cleaned_path, cleaned)
    print(f'Wrote cleaned ASR -> {cleaned_path} ({len(cleaned)} items)')

    # label using teacher_labeler if faces provided
    if faces and faces.exists():
        tracks = load_face_tracks_jsonl(str(faces))
        labeled = label_asr_segments(cleaned, tracks)
        labeled_path = out_prefix.with_suffix('.labeled.cleaned.jsonl')
        write_jsonl(labeled_path, labeled)
        print(f'Wrote labeled cleaned -> {labeled_path} ({len(labeled)} items)')


if __name__ == '__main__':
    main()
