"""Run the live ASR test but inject a fake pyaudio module (sends silent frames).

This allows testing DashScope connectivity without a real microphone.
"""
import sys
import types
from pathlib import Path
from dotenv import load_dotenv
import runpy

# Load .env
proj_root = Path(__file__).resolve().parents[1]
load_dotenv(str(proj_root / '.env'))
# Ensure project root on sys.path
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

# Inject fake pyaudio
fake_pyaudio = types.ModuleType('pyaudio')

class FakeStream:
    def __init__(self, frames_per_buffer=3200):
        self._open = True

    def read(self, n, exception_on_overflow=False):
        # return silent PCM16 (n samples -> 2*n bytes)
        return b"\x00" * (n * 2)

    def stop_stream(self):
        self._open = False

    def close(self):
        self._open = False

class FakePyAudio:
    def __init__(self):
        pass

    def open(self, format=None, channels=1, rate=16000, input=True, frames_per_buffer=3200):
        return FakeStream(frames_per_buffer=frames_per_buffer)

    def terminate(self):
        pass

fake_pyaudio.PyAudio = FakePyAudio
fake_pyaudio.paInt16 = 8
sys.modules['pyaudio'] = fake_pyaudio

# Some projects expect DASHSCOPE_API_KEY (no underscore) — copy if alternate name exists
import os
if os.getenv('DASH_SCOPE_API_KEY') and not os.getenv('DASHSCOPE_API_KEY'):
    os.environ['DASHSCOPE_API_KEY'] = os.environ['DASH_SCOPE_API_KEY']

# If real `dashscope` is not installed, inject a minimal fake that provides
# `dashscope.audio.asr.Recognition` and `RecognitionCallback` so the client can run.
try:
    import dashscope  # type: ignore
except Exception:
    fake_pkg = types.ModuleType('dashscope')
    fake_audio = types.ModuleType('dashscope.audio')
    fake_asr = types.ModuleType('dashscope.audio.asr')

    class FakeRecognition:
        def __init__(self, model=None, format=None, sample_rate=16000, callback=None):
            self.callback = callback
            self._running = False
        def start(self):
            # call on_open then run a background thread to emit nothing (we use fake pyaudio)
            try:
                if hasattr(self.callback, 'on_open'):
                    self.callback.on_open()
            except Exception:
                pass
        def send_audio_frame(self, data: bytes):
            # ignore audio frames
            pass
        def stop(self):
            try:
                if hasattr(self.callback, 'on_close'):
                    self.callback.on_close()
            except Exception:
                pass

    class FakeRecognitionCallback:
        pass

    fake_asr.Recognition = FakeRecognition
    fake_asr.RecognitionCallback = FakeRecognitionCallback
    fake_asr.RecognitionResult = type('RecognitionResult', (), {})

    sys.modules['dashscope'] = fake_pkg
    sys.modules['dashscope.audio'] = fake_audio
    sys.modules['dashscope.audio.asr'] = fake_asr

# Run the live test script (it will read DASH_SCOPE_API_KEY from .env)
runpy.run_path(str(proj_root / 'tools' / 'test_asr_client_live.py'), run_name='__main__')
