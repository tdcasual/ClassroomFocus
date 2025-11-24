import json
from pathlib import Path
from typing import Dict, Optional

import cv2


class Recorder:
    """
    Lightweight recorder for video frames and JSONL event logs.

    Example:
        rec = Recorder(video_path="out.mp4",
                       cv_event_path="cv_events.jsonl",
                       asr_event_path="asr_events.jsonl",
                       synced_event_path="synced_events.jsonl")
        rec.open(frame_width=640, frame_height=480, fps=30)
        rec.write_frame(frame)
        rec.log_cv_event(ev)
        rec.close()
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        cv_event_path: Optional[str] = None,
        asr_event_path: Optional[str] = None,
        synced_event_path: Optional[str] = None,
    ):
        self.video_path = Path(video_path) if video_path else None
        self.audio_path = Path(audio_path) if audio_path else None
        self.cv_event_path = Path(cv_event_path) if cv_event_path else None
        self.asr_event_path = Path(asr_event_path) if asr_event_path else None
        self.synced_event_path = Path(synced_event_path) if synced_event_path else None

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._audio_fh = None
        self._cv_fh = None
        self._asr_fh = None
        self._synced_fh = None

    def open(self, frame_width: int, frame_height: int, fps: int = 30) -> None:
        """Initialize file handles and video writer; call once before logging."""
        if self.video_path:
            self._ensure_parent(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(self.video_path), fourcc, fps, (frame_width, frame_height)
            )
            if not self._video_writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {self.video_path}")
        if self.audio_path:
            self._ensure_parent(self.audio_path)
            self._audio_fh = self.audio_path.open("ab")
        if self.cv_event_path:
            self._ensure_parent(self.cv_event_path)
            self._cv_fh = self.cv_event_path.open("a", encoding="utf-8")
        if self.asr_event_path:
            self._ensure_parent(self.asr_event_path)
            self._asr_fh = self.asr_event_path.open("a", encoding="utf-8")
        if self.synced_event_path:
            self._ensure_parent(self.synced_event_path)
            self._synced_fh = self.synced_event_path.open("a", encoding="utf-8")

    def write_frame(self, frame) -> None:
        """Write one video frame; ignored if video writer not configured."""
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def write_audio(self, audio_bytes: bytes) -> None:
        """Append raw audio bytes if audio logging is enabled."""
        if self._audio_fh is not None:
            self._audio_fh.write(audio_bytes)

    def log_cv_event(self, ev: Dict) -> None:
        if self._cv_fh is not None:
            self._write_json_line(self._cv_fh, ev)

    def log_asr_event(self, ev: Dict) -> None:
        if self._asr_fh is not None:
            self._write_json_line(self._asr_fh, ev)

    def log_synced_event(self, ev: Dict) -> None:
        if self._synced_fh is not None:
            self._write_json_line(self._synced_fh, ev)

    def close(self) -> None:
        """Release all resources."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        for fh in (self._audio_fh, self._cv_fh, self._asr_fh, self._synced_fh):
            try:
                if fh is not None:
                    fh.flush()
                    fh.close()
            except Exception:
                pass
        self._audio_fh = None
        self._cv_fh = None
        self._asr_fh = None
        self._synced_fh = None

    # ---- helpers ----
    @staticmethod
    def _write_json_line(fh, obj: Dict) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
