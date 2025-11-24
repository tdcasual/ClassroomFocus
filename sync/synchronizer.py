import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class SyncedEvent:
    ts: float
    student_id: int
    cv_type: str
    cv_dur: Optional[float]
    ear: Optional[float]
    pitch: Optional[float]
    asr_text: Optional[str]
    asr_ts: Optional[float] = None


class DataSynchronizer:
    """
    Aligns CV events (drowsy/down/blink) with ASR sentence events on a shared time axis.

    Usage:
        sync = DataSynchronizer(align_window=3.0)
        sync.add_cv_events(cv_events)   # list of dicts from FaceAnalyzer
        sync.add_asr_event(asr_event)   # {"ts": float, "type": "ASR_SENTENCE", "text": str}
        synced = sync.get_synced_events()
    """

    def __init__(self, align_window: float = 3.0):
        self.align_window = align_window
        self.cv_events: Deque[Dict] = deque()
        self.asr_events: Deque[Dict] = deque()
        self.synced_events: Deque[SyncedEvent] = deque()
        self._lock = threading.Lock()

    def add_cv_events(self, events: List[Dict]) -> None:
        """Add a batch of CV events; each will be aligned to the closest ASR sentence."""
        if not events:
            return
        with self._lock:
            for ev in events:
                if "ts" not in ev:
                    continue
                self.cv_events.append(ev)
                self._align_new_cv_event(ev)
            self._trim_buffers()

    def add_asr_event(self, event: Dict) -> None:
        """Add a single ASR event; kept for alignment when future CV events arrive."""
        if not event or "ts" not in event:
            return
        with self._lock:
            self.asr_events.append(event)
            self._align_asr_to_cv(event)
            self._trim_buffers()

    def get_synced_events(self) -> List[Dict]:
        """Return and clear all synced events accumulated so far."""
        with self._lock:
            out = [se.__dict__ for se in self.synced_events]
            self.synced_events.clear()
            return out

    # ---- internal helpers ----
    def _align_new_cv_event(self, cv_ev: Dict) -> None:
        """Align one CV event to the nearest ASR sentence inside align_window."""
        best_asr = None
        best_dt = self.align_window + 1.0
        for asr_ev in self.asr_events:
            if asr_ev.get("type") != "ASR_SENTENCE":
                continue
            dt = abs(cv_ev["ts"] - asr_ev["ts"])
            if dt < best_dt:
                best_dt = dt
                best_asr = asr_ev
        if best_asr is None or best_dt > self.align_window:
            synced = SyncedEvent(
                ts=cv_ev["ts"],
                student_id=cv_ev.get("student_id", -1),
                cv_type=cv_ev.get("type", "UNKNOWN"),
                cv_dur=cv_ev.get("dur"),
                ear=cv_ev.get("ear"),
                pitch=cv_ev.get("pitch"),
                asr_text=None,
                asr_ts=None,
            )
            self.synced_events.append(synced)
            cv_ev["_asr_linked"] = True
            return

        synced = SyncedEvent(
            ts=cv_ev["ts"],
            student_id=cv_ev.get("student_id", -1),
            cv_type=cv_ev.get("type", "UNKNOWN"),
            cv_dur=cv_ev.get("dur"),
            ear=cv_ev.get("ear"),
            pitch=cv_ev.get("pitch"),
            asr_text=best_asr.get("text"),
            asr_ts=best_asr.get("ts"),
        )
        self.synced_events.append(synced)
        cv_ev["_asr_linked"] = True

    def _align_asr_to_cv(self, asr_ev: Dict) -> None:
        """Align a newly arrived ASR sentence to any recent CV events not yet linked."""
        if asr_ev.get("type") != "ASR_SENTENCE":
            return
        asr_ts = asr_ev["ts"]
        for cv_ev in reversed(self.cv_events):
            if cv_ev.get("_asr_linked"):
                continue
            dt = abs(asr_ts - cv_ev["ts"])
            if dt > self.align_window:
                continue
            synced = SyncedEvent(
                ts=cv_ev["ts"],
                student_id=cv_ev.get("student_id", -1),
                cv_type=cv_ev.get("type", "UNKNOWN"),
                cv_dur=cv_ev.get("dur"),
                ear=cv_ev.get("ear"),
                pitch=cv_ev.get("pitch"),
                asr_text=asr_ev.get("text"),
                asr_ts=asr_ts,
            )
            self.synced_events.append(synced)
            cv_ev["_asr_linked"] = True

    def _trim_buffers(self) -> None:
        """Drop old events outside the window based on latest ts in either stream."""
        if not self.cv_events and not self.asr_events:
            return
        latest_ts = max(
            self.cv_events[-1]["ts"] if self.cv_events else 0.0,
            self.asr_events[-1]["ts"] if self.asr_events else 0.0,
        )
        while self.cv_events and latest_ts - self.cv_events[0]["ts"] > self.align_window:
            self.cv_events.popleft()
        while self.asr_events and latest_ts - self.asr_events[0]["ts"] > self.align_window:
            self.asr_events.popleft()
