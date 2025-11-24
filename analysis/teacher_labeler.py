import json
from typing import List, Dict, Any, Optional


def _parse_time(ev: Dict[str, Any]) -> Optional[Dict[str, float]]:
    start_keys = ["start_time", "start", "begin", "ts", "t"]
    end_keys = ["end_time", "end", "stop"]
    start = None
    end = None
    for k in start_keys:
        if k in ev:
            try:
                start = float(ev[k])
            except Exception:
                start = None
            break
    for k in end_keys:
        if k in ev:
            try:
                end = float(ev[k])
            except Exception:
                end = None
            break
    if start is None:
        return None
    if end is None:
        if "duration" in ev:
            try:
                end = start + float(ev["duration"])
            except Exception:
                end = start + 1.0
        else:
            end = start + 1.0
    return {"start": start, "end": end}


def load_asr_jsonl(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            times = _parse_time(ev)
            text = ev.get("text") or ev.get("transcript") or ev.get("result") or ev.get("sentence") or ev.get("raw_text")
            if times is None:
                ts = ev.get("ts") or ev.get("timestamp")
                if ts is not None:
                    try:
                        times = {"start": float(ts), "end": float(ts) + float(ev.get("duration", 1.0))}
                    except Exception:
                        continue
                else:
                    continue
            events.append({"start": float(times["start"]), "end": float(times["end"]), "text": text or "", "raw": ev})
    return events


def load_face_tracks_jsonl(path: str) -> Dict[str, List[Dict[str, Any]]]:
    tracks: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = ev.get("ts") or ev.get("timestamp") or ev.get("t")
            track_id = ev.get("track_id") or ev.get("id") or ev.get("face_id")
            bbox = ev.get("bbox") or ev.get("rect")
            if ts is None or track_id is None:
                continue
            try:
                tsv = float(ts)
            except Exception:
                continue
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    area = max(0.0, w * h)
                    cx = x + w / 2.0
                except Exception:
                    area = float(ev.get("area", 0.0))
                    cx = float(ev.get("center_x", 0.5))
            else:
                area = float(ev.get("area", 0.0))
                cx = float(ev.get("center_x", 0.5))
            rec = {"ts": tsv, "area": area, "center_x": cx, "raw": ev}
            tracks.setdefault(str(track_id), []).append(rec)
    for k in list(tracks.keys()):
        tracks[k].sort(key=lambda r: r["ts"])
    return tracks


def choose_teacher_track(tracks: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    if not tracks:
        return None
    stats = {}
    areas = []
    for tid, recs in tracks.items():
        presence = len(recs)
        avg_area = sum(r.get("area", 0.0) for r in recs) / max(1, presence)
        avg_cx = sum(r.get("center_x", 0.5) for r in recs) / max(1, presence)
        stats[tid] = {"presence": presence, "avg_area": avg_area, "avg_cx": avg_cx}
        areas.append(avg_area)
    max_area = max(areas) if areas else 1.0
    best_tid = None
    best_score = -1.0
    for tid, s in stats.items():
        norm_area = s["avg_area"] / max_area if max_area > 0 else 0.0
        center_dist = abs((s["avg_cx"]) - 0.5)
        score = s["presence"] * (1.0 + norm_area) * (1.0 - center_dist)
        if score > best_score:
            best_score = score
            best_tid = tid
    return best_tid


def label_asr_segments(asr_events: List[Dict[str, Any]], face_tracks: Dict[str, List[Dict[str, Any]]], tolerance: float = 0.3, presence_fraction: float = 0.01) -> List[Dict[str, Any]]:
    """
    Label ASR segments as `teacher` or `student` using Option B heuristics.

    Parameters:
    - tolerance: time window (seconds) to expand segment when checking for teacher presence (helps alignment errors)
    - presence_fraction: minimum fraction of timestamps within the expanded segment for teacher considered present
    """
    labels: List[Dict[str, Any]] = []
    teacher_tid = choose_teacher_track(face_tracks)
    for seg in asr_events:
        start = float(seg["start"])
        end = float(seg["end"])
        label = "student"
        if teacher_tid is not None:
            recs = face_tracks.get(teacher_tid, [])
            if recs:
                window_start = start - tolerance
                window_end = end + tolerance
                total = 0
                inside = 0
                for r in recs:
                    total += 1
                    if r["ts"] >= window_start and r["ts"] <= window_end:
                        inside += 1
                frac = (inside / total) if total > 0 else 0.0
                if frac >= presence_fraction:
                    label = "teacher"
        labels.append({"start": start, "end": end, "text": seg.get("text", ""), "label": label, "raw": seg.get("raw")})
    return labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", required=True)
    parser.add_argument("--faces", required=False)
    args = parser.parse_args()
    asr = load_asr_jsonl(args.asr)
    faces = load_face_tracks_jsonl(args.faces) if args.faces else {}
    labeled = label_asr_segments(asr, faces)
    print(json.dumps(labeled, ensure_ascii=False, indent=2))
