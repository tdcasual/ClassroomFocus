import json as _json

import pytest

from tools.openai_compat import (
    OpenAICompat,
    OpenAICompatConfig,
    _extract_json_object,
    _normalize_v1_base_url,
)


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = int(status_code)
        self._payload = payload
        try:
            self.text = _json.dumps(payload, ensure_ascii=False)
        except Exception:
            self.text = str(payload)

    def json(self):
        return self._payload


def test_normalize_v1_base_url():
    assert _normalize_v1_base_url("https://api.openai.com") == "https://api.openai.com/v1"
    assert _normalize_v1_base_url("https://api.openai.com/") == "https://api.openai.com/v1"
    assert _normalize_v1_base_url("https://example.com/v1") == "https://example.com/v1"
    assert _normalize_v1_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_generate_text_prefers_responses(monkeypatch):
    client = OpenAICompat(OpenAICompatConfig(base_url="https://example.com", api_key="k", model="m"))
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json})
        assert url.endswith("/v1/responses")
        assert "max_output_tokens" in (json or {})
        return _Resp(200, {"output": [{"content": [{"type": "output_text", "text": "hello"}]}]})

    monkeypatch.setattr("tools.openai_compat.requests.post", fake_post)
    txt = client.generate_text(messages=[{"role": "user", "content": "hi"}], max_tokens=12)
    assert txt == "hello"
    assert len(calls) == 1


def test_generate_text_fallback_to_chat_completions(monkeypatch):
    client = OpenAICompat(OpenAICompatConfig(base_url="https://example.com", api_key="k", model="m"))
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json})
        if url.endswith("/v1/responses"):
            return _Resp(404, {"error": {"message": "no responses"}})
        if url.endswith("/v1/chat/completions"):
            assert "messages" in (json or {})
            assert "max_tokens" in (json or {})
            return _Resp(200, {"choices": [{"message": {"content": "ok-from-chat"}}]})
        return _Resp(500, {"error": {"message": "unexpected"}})

    monkeypatch.setattr("tools.openai_compat.requests.post", fake_post)
    txt = client.generate_text(messages=[{"role": "user", "content": "hi"}], max_tokens=12)
    assert txt == "ok-from-chat"
    # 3 retries for /responses + 1 call to /chat/completions
    assert [c["url"].split("/")[-1] for c in calls] == ["responses", "responses", "responses", "completions"]


def test_generate_text_chat_max_completion_tokens_retry(monkeypatch):
    client = OpenAICompat(OpenAICompatConfig(base_url="https://example.com", api_key="k", model="m"))
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json})
        if url.endswith("/v1/responses"):
            return _Resp(404, {"error": {"message": "no responses"}})
        if url.endswith("/v1/chat/completions"):
            # pretend backend only accepts `max_completion_tokens`
            if "max_tokens" in (json or {}):
                return _Resp(400, {"error": {"message": "use max_completion_tokens"}})
            assert "max_completion_tokens" in (json or {})
            return _Resp(200, {"choices": [{"message": {"content": "ok-retry"}}]})
        return _Resp(500, {"error": {"message": "unexpected"}})

    monkeypatch.setattr("tools.openai_compat.requests.post", fake_post)
    txt = client.generate_text(messages=[{"role": "user", "content": "hi"}], max_tokens=12)
    assert txt == "ok-retry"
    # 3 /responses attempts + 2 /chat attempts
    assert len(calls) == 5
    assert calls[-1]["url"].endswith("/v1/chat/completions")
    assert "max_completion_tokens" in calls[-1]["json"]


def test_extract_json_object_is_robust():
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    assert _extract_json_object("prefix {\"a\": 1} suffix") == {"a": 1}
    assert _extract_json_object("no json here") is None


def test_transcribe_audio_smoke(monkeypatch, tmp_path):
    client = OpenAICompat(OpenAICompatConfig(base_url="https://example.com", api_key="k", model="m"))
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")  # minimal placeholder; server is mocked
    calls = []

    def fake_post(url, headers=None, data=None, files=None, timeout=None, **kwargs):
        calls.append({"url": url, "data": data, "files": files})
        assert url.endswith("/v1/audio/transcriptions")
        assert data["response_format"] == "verbose_json"
        assert data["timestamp_granularities[]"] == "segment"
        return _Resp(200, {"text": "hello", "segments": [{"start": 0, "end": 1, "text": "hello"}]})

    monkeypatch.setattr("tools.openai_compat.requests.post", fake_post)
    out = client.transcribe_audio(str(audio_path))
    assert out["text"] == "hello"
    assert calls and calls[0]["files"]["file"][0] == "a.wav"


def test_transcribe_audio_error(monkeypatch, tmp_path):
    client = OpenAICompat(OpenAICompatConfig(base_url="https://example.com", api_key="k", model="m"))
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"x")

    def fake_post(url, headers=None, data=None, files=None, timeout=None, **kwargs):
        return _Resp(401, {"error": {"message": "bad key"}})

    monkeypatch.setattr("tools.openai_compat.requests.post", fake_post)
    with pytest.raises(RuntimeError, match="bad key"):
        client.transcribe_audio(str(audio_path))

