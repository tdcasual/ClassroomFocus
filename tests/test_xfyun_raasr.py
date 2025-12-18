import base64
import hashlib
import json


class _FakeResp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def test_slice_id_base26():
    from asr.xfyun_raasr import _slice_id

    assert _slice_id(0) == "aaaaaaaa"
    assert _slice_id(1) == "aaaaaaab"
    assert _slice_id(25) == "aaaaaaaz"
    assert _slice_id(26) == "aaaaaaba"


def test_signa_md5_base64():
    from asr.xfyun_raasr import _signa

    app_id = "appid"
    ts = "1700000000"
    secret = "secret"
    expected = base64.b64encode(hashlib.md5((app_id + ts + secret).encode("utf-8")).digest()).decode("utf-8")
    assert _signa(app_id, ts, secret) == expected


def test_parse_result_data_seconds():
    from asr.xfyun_raasr import _parse_result_data

    data_list = [
        json.dumps({"bg": 1000, "ed": 2200, "onebest": "hello"}),
        json.dumps({"bg": 0, "ed": 500, "onebest": "hi"}),
    ]
    segs = _parse_result_data(data_list)
    assert [s["text"] for s in segs] == ["hi", "hello"]
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 0.5
    assert segs[1]["start"] == 1.0
    assert segs[1]["end"] == 2.2


def test_transcribe_wav_upload_uses_multipart_files(tmp_path, monkeypatch):
    import asr.xfyun_raasr as mod

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"not-a-real-wav-but-nonempty")

    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append({"url": str(url), "data": dict(data or {}), "files": files, "timeout": timeout})
        if str(url).endswith("/prepare"):
            return _FakeResp(200, {"ok": 0, "data": "task123"})
        if str(url).endswith("/upload"):
            return _FakeResp(200, {"ok": 0})
        if str(url).endswith("/merge"):
            return _FakeResp(200, {"ok": 0})
        if str(url).endswith("/getResult"):
            return _FakeResp(200, {"ok": 0, "data": [json.dumps({"bg": 0, "ed": 1200, "onebest": "hi"})]})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(mod.requests, "post", fake_post)

    cfg = mod.XfyunRaasrConfig(
        app_id="appid",
        secret_key="secret",
        host="https://raasr.xfyun.cn/v2/api",
        slice_size_bytes=10**9,
        poll_interval_sec=0.0,
        timeout_sec=2.0,
    )
    segs, meta = mod.transcribe_wav(str(audio), cfg)
    assert segs and segs[0]["text"] == "hi"
    assert meta.get("task_id") == "task123"

    upload = [c for c in calls if c["url"].endswith("/upload")]
    assert len(upload) == 1
    assert upload[0]["files"] and "content" in upload[0]["files"]
    assert upload[0]["files"]["content"][0] == "slice"
    assert isinstance(upload[0]["files"]["content"][1], (bytes, bytearray))
