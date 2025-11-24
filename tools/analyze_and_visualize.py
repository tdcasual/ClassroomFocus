"""
CLI to run analysis (teacher labeling) and visualization.

Example:
  python tools/analyze_and_visualize.py --asr logs/asr_events.jsonl --faces logs/face_tracks.jsonl --out-prefix out/session1

Outputs:
  - {out-prefix}.labeled.jsonl
  - {out-prefix}.timeline.png
"""
import os
from pathlib import Path

from analysis.teacher_labeler import load_asr_jsonl, load_face_tracks_jsonl, label_asr_segments
from viz.visualizer import plot_timeline, save_labeled_jsonl


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--asr", required=True, help="ASR JSONL input path")
  parser.add_argument("--faces", required=False, help="Face tracks JSONL input path")
  parser.add_argument("--out-prefix", required=True, help="Output prefix for files")
  parser.add_argument("--tolerance", type=float, default=0.3, help="time tolerance (s) when matching ASR to face tracks")
  parser.add_argument("--presence-fraction", type=float, default=0.01, help="min fraction of face samples to consider teacher present")
  parser.add_argument("--show-text", action='store_true', help="Show ASR text on timeline (may require font support)")
  parser.add_argument("--font-path", type=str, default=None, help="Path to a TTF/OTF font to render text (avoids missing glyphs)")
  args = parser.parse_args()

  asr_events = load_asr_jsonl(args.asr)
  face_tracks = load_face_tracks_jsonl(args.faces) if args.faces else {}
  labeled = label_asr_segments(asr_events, face_tracks, tolerance=args.tolerance, presence_fraction=args.presence_fraction)

  out_prefix = Path(args.out_prefix)
  out_prefix.parent.mkdir(parents=True, exist_ok=True)

  labeled_jsonl = str(out_prefix) + ".labeled.jsonl"
  timeline_png = str(out_prefix) + ".timeline.png"

  save_labeled_jsonl(labeled, labeled_jsonl)
  plot_timeline(labeled, timeline_png, title="ASR Teacher/Student Timeline", face_tracks=face_tracks, show_text=args.show_text, font_path=args.font_path)

  print("Wrote:", labeled_jsonl)
  print("Wrote:", timeline_png)


if __name__ == "__main__":
    main()
