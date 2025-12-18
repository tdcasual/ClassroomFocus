from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover
    pyaudio = None
try:
    import dashscope  # type: ignore
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult  # type: ignore
except Exception:  # pragma: no cover
    dashscope = None
    Recognition = None  # type: ignore

    class RecognitionCallback:  # type: ignore
        pass

    class RecognitionResult:  # type: ignore
        pass


class AliASRClient:
    """
    Ali DashScope streaming ASR client.

    on_sentence callback receives:
        {"ts": float, "type": "ASR_SENTENCE", "text": str, "raw": dict}
    where ts uses the shared relative time base provided by `time_base()`.
    """

    def __init__(
        self,
        api_key: Optional[str],
        time_base: Callable[[], float],
        on_sentence: Callable[[dict], None],
        model: str = "fun-asr-realtime",
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        auto_open_mic: bool = True,
    ):
        self.api_key = api_key
        self.time_base = time_base
        self.on_sentence = on_sentence
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms

        self._stream = None
        self._mic = None
        self._recog: Optional[Recognition] = None
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.auto_open_mic = auto_open_mic

    def start(self) -> None:
        """Start ASR in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        if dashscope is None or Recognition is None:
            raise RuntimeError("dashscope is required for AliASRClient (pip install dashscope).")
        if self.auto_open_mic and pyaudio is None:
            raise RuntimeError("pyaudio is required for streaming mic capture (pip install pyaudio).")
        if self.api_key:
            dashscope.api_key = self.api_key
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal stop and wait for thread exit."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- internal ----
    def _run_loop(self):
        if Recognition is None:
            raise RuntimeError("dashscope Recognition is not available.")
        callback = _AliCallback(self)
        self._recog = Recognition(
            model=self.model,
            format="pcm",
            sample_rate=self.sample_rate,
            callback=callback,
        )
        try:
            self._recog.start()
            chunk_bytes = int(self.sample_rate * (self.chunk_ms / 1000.0) * 2)  # int16 mono
            while not self._stop_flag.is_set():
                if self._stream:
                    data = self._stream.read(chunk_bytes // 2, exception_on_overflow=False)
                    self._recog.send_audio_frame(data)
                else:
                    time.sleep(0.01)
        finally:
            try:
                self._recog.stop()
            except Exception:
                pass
            callback.cleanup()
            self._recog = None


class _AliCallback(RecognitionCallback):
    """Wraps RecognitionCallback to pipe sentences into user callback."""

    def __init__(self, outer: AliASRClient):
        super().__init__()
        self.outer = outer

    def on_open(self) -> None:
        # Only auto-open microphone stream if allowed. Otherwise the outer
        # client may be used in file/offline mode and will feed audio frames
        # directly to the Recognition instance.
        if self.outer._mic or self.outer._stream:
            return
        if not getattr(self.outer, "auto_open_mic", True):
            return
        if pyaudio is None:
            raise RuntimeError("pyaudio is required to open microphone stream.")
        self.outer._mic = pyaudio.PyAudio()
        self.outer._stream = self.outer._mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.outer.sample_rate,
            input=True,
            frames_per_buffer=int(self.outer.sample_rate * (self.outer.chunk_ms / 1000.0)),
        )

    def on_close(self) -> None:
        self.cleanup()

    def on_event(self, result: RecognitionResult) -> None:
        text = result.get_sentence()
        if not text:
            return
        ts = float(self.outer.time_base())
        ev = {"ts": ts, "type": "ASR_SENTENCE", "text": text, "raw": result.__dict__}
        try:
            self.outer.on_sentence(ev)
        except Exception:
            # Swallow callback errors to avoid breaking the ASR loop
            pass

    def cleanup(self):
        if self.outer._stream:
            try:
                self.outer._stream.stop_stream()
                self.outer._stream.close()
            except Exception:
                pass
        if self.outer._mic:
            try:
                self.outer._mic.terminate()
            except Exception:
                pass
        self.outer._stream = None
        self.outer._mic = None


    
def transcribe_file(api_key: Optional[str], wav_path: str, time_base: Callable[[], float], on_sentence: Callable[[dict], None], model: str = "fun-asr-realtime", sample_rate: int = 16000, chunk_ms: int = 100, progress_hook: Optional[Callable[[float], None]] = None):
    """Utility to transcribe a WAV file by sending chunks to Recognition.

    This function does not open any pyaudio streams and is safe to run when
    the environment has no audio input device. It will call `on_sentence`
    with the same event structure as the live client.
    """
    if dashscope is None or Recognition is None:
        raise RuntimeError("dashscope is required for transcribe_file (pip install dashscope).")
    if api_key:
        dashscope.api_key = api_key
    # Create a fresh Recognition + callback that routes to provided on_sentence
    class _LocalCB(RecognitionCallback):
        def on_open(self):
            return

        def on_close(self):
            return

        def on_event(self, result: 'RecognitionResult') -> None:
            try:
                txt = result.get_sentence()
            except Exception:
                txt = None
            if not txt:
                return
            ts = float(time_base())
            ev = {"ts": ts, "type": "ASR_SENTENCE", "text": txt, "raw": result.__dict__}
            try:
                on_sentence(ev)
            except Exception:
                pass

    recog = Recognition(model=model, format="pcm", sample_rate=sample_rate, callback=_LocalCB())
    recog.start()
    frames_sent = 0
    try:
        import wave as _wave
        with _wave.open(wav_path, 'rb') as wf:
            bytes_per_frame = wf.getsampwidth() * wf.getnchannels()
            frames_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
            while True:
                chunk = wf.readframes(frames_per_chunk)
                if not chunk:
                    break
                recog.send_audio_frame(chunk)
                if bytes_per_frame > 0:
                    frames_sent += len(chunk) // bytes_per_frame
                    if progress_hook:
                        try:
                            progress_hook(frames_sent / float(sample_rate))
                        except Exception:
                            pass
    finally:
        try:
            recog.stop()
        except Exception:
            pass
    if progress_hook:
        try:
            progress_hook(frames_sent / float(sample_rate))
        except Exception:
            pass
