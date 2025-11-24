"""Mock ASR client test that does NOT require dashscope or network.

This script fakes the `Recognition` behavior by injecting a small module into
`sys.modules` before importing `asr.asr_client`, so the `AliASRClient` can be
imported even if `dashscope` isn't installed.

It then starts the client and the fake Recognition will emit a few fake
"sentences" via the callback to validate the pipeline and resource cleanup.
"""
import sys
import time
import threading
import types
from typing import Any, Dict
from pathlib import Path

# Ensure project root is on sys.path so `import asr.asr_client` works
proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Create a fake dashscope.audio.asr module with Recognition, RecognitionCallback classes
fake_asr_mod = types.SimpleNamespace()

class FakeRecognition:
    def __init__(self, model=None, format=None, sample_rate=16000, callback=None):
        self.callback = callback
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        # start thread to emit fake results
        def run():
            # wait for on_open to be processed (as real Recognition would call it)
            try:
                if hasattr(self.callback, 'on_open'):
                    self.callback.on_open()
            except Exception:
                pass
            # emit a few fake sentences
            for i in range(3):
                if not self._running:
                    break
                time.sleep(1.0)
                # create a fake result object with get_sentence()
                class R:
                    def __init__(self, t):
                        self._t = t
                    def get_sentence(self):
                        return self._t
                r = R(f'fake sentence {i+1}')
                try:
                    if hasattr(self.callback, 'on_event'):
                        self.callback.on_event(r)
                except Exception:
                    pass
            # simulate close
            try:
                if hasattr(self.callback, 'on_close'):
                    self.callback.on_close()
            except Exception:
                pass
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def send_audio_frame(self, data: bytes):
        # ignore
        pass

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

# Placeholder callback base class
class FakeRecognitionCallback:
    pass

# Build nested module structure in sys.modules
fake_pkg = types.ModuleType('dashscope')
fake_audio = types.ModuleType('dashscope.audio')
fake_asr = types.ModuleType('dashscope.audio.asr')
fake_asr.Recognition = FakeRecognition
fake_asr.RecognitionCallback = FakeRecognitionCallback
fake_asr.RecognitionResult = type('FakeRecognitionResult', (), {})
# push into sys.modules
sys.modules['dashscope'] = fake_pkg
sys.modules['dashscope.audio'] = fake_audio
sys.modules['dashscope.audio.asr'] = fake_asr

# Provide a minimal fake `pyaudio` so `asr.asr_client` can import and open a dummy stream
class FakeStream:
    def __init__(self, frames_per_buffer=3200):
        self._fpb = frames_per_buffer
        self._open = True

    def read(self, n, exception_on_overflow=False):
        # return silence (int16 -> 2 bytes per sample)
        return b"\x00" * n * 2

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

fake_pyaudio = types.ModuleType('pyaudio')
fake_pyaudio.PyAudio = FakePyAudio
fake_pyaudio.paInt16 = 8
sys.modules['pyaudio'] = fake_pyaudio

# Now import the real client (it will pick up our fake dashscope)
from asr.asr_client import AliASRClient


def time_base():
    return time.time()


def on_sentence(ev: Dict[str, Any]):
    print('MOCK ASR EVENT:', ev)


if __name__ == '__main__':
    client = AliASRClient(api_key=None, time_base=time_base, on_sentence=on_sentence)
    print('Starting mock ASR client...')
    client.start()
    # Wait enough time for fake sentences to be emitted
    try:
        time.sleep(5.0)
    except KeyboardInterrupt:
        pass
    print('Stopping mock ASR client...')
    client.stop()
    print('Done')
