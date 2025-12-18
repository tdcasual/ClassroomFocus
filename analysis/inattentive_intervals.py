from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    kind: str


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def normalize_intervals(raw: Iterable[Dict[str, Any]]) -> List[Interval]:
    out: List[Interval] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        s = _to_float(it.get("start"))
        e = _to_float(it.get("end"))
        if s is None or e is None:
            continue
        if e <= s:
            continue
        kind = str(it.get("type") or it.get("kind") or "").strip().upper() or "UNKNOWN"
        out.append(Interval(start=float(s), end=float(e), kind=kind))
    out.sort(key=lambda x: (x.start, x.end, x.kind))
    return out


def infer_not_visible_intervals(
    face_times: Sequence[float],
    session_end: float,
    gap_sec: float = 1.5,
    tail_cap_sec: Optional[float] = 12.0,
) -> List[Dict[str, Any]]:
    """Infer 'NOT_VISIBLE' intervals from gaps in face observations.

    `face_times` are timestamps when a face track is present (has landmarks).
    Any gap >= `gap_sec` is treated as "eyes not visible" (e.g. head down).
    """
    gap_sec = float(gap_sec)
    if gap_sec <= 0:
        return []
    end_ts = _to_float(session_end)
    if end_ts is None or end_ts <= 0:
        end_ts = 0.0

    ts = sorted({float(t) for t in face_times if _to_float(t) is not None})
    if not ts:
        return []

    out: List[Dict[str, Any]] = []
    prev = ts[0]
    for cur in ts[1:]:
        if (cur - prev) >= gap_sec:
            out.append({"type": "NOT_VISIBLE", "start": float(prev), "end": float(cur)})
        prev = cur

    if end_ts > 0 and (end_ts - ts[-1]) >= gap_sec:
        tail_end = end_ts
        if tail_cap_sec is not None:
            try:
                cap = float(tail_cap_sec)
            except Exception:
                cap = 0.0
            if cap > 0:
                tail_end = min(tail_end, ts[-1] + cap)
        if tail_end > ts[-1]:
            out.append({"type": "NOT_VISIBLE", "start": float(ts[-1]), "end": float(tail_end)})
    return out


def merge_inattentive_intervals(
    intervals: Iterable[Dict[str, Any]],
    join_gap_sec: float = 0.3,
    out_type: str = "INATTENTIVE",
) -> List[Dict[str, Any]]:
    """Merge heterogeneous intervals into union intervals.

    Output intervals include:
      - start/end (float)
      - type (default 'INATTENTIVE')
      - kinds: list[str] of contributing interval types
    """
    join_gap = float(join_gap_sec)
    join_gap = max(0.0, join_gap)

    xs = normalize_intervals(intervals)
    if not xs:
        return []

    merged: List[Tuple[float, float, set[str]]] = []
    cur_s = xs[0].start
    cur_e = xs[0].end
    kinds = {xs[0].kind}

    for it in xs[1:]:
        if it.start <= (cur_e + join_gap):
            cur_e = max(cur_e, it.end)
            kinds.add(it.kind)
            continue
        merged.append((cur_s, cur_e, set(kinds)))
        cur_s, cur_e, kinds = it.start, it.end, {it.kind}

    merged.append((cur_s, cur_e, set(kinds)))

    out: List[Dict[str, Any]] = []
    for s, e, ks in merged:
        if e <= s:
            continue
        out.append({"type": out_type, "start": float(s), "end": float(e), "kinds": sorted(ks)})
    return out


def union_duration(intervals: Iterable[Dict[str, Any]]) -> float:
    """Compute total duration of merged union intervals (seconds)."""
    total = 0.0
    for it in normalize_intervals(intervals):
        total += max(0.0, it.end - it.start)
    return float(total)

