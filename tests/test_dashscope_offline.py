def test_extract_ms_supports_start_time_end_time():
    from asr.dashscope_offline import _extract_ms

    raw = {"start_time": 1000, "end_time": 2500}
    ms = _extract_ms(raw, keys=["begin_time", "begin", "start_time", "start"])
    assert ms == (1000.0, 2500.0)


def test_dashscope_offline_segments_fallback_and_timestamps(monkeypatch):
    from asr.dashscope_offline import DashScopeOfflineConfig, transcribe_wav_to_segments

    # Patch the DashScope transcribe_file to avoid requiring dashscope + network.
    import asr.asr_client as asr_client

    def fake_transcribe_file(**kwargs):
        progress_hook = kwargs.get("progress_hook")
        on_sentence = kwargs.get("on_sentence")

        progress_hook(1.0)
        on_sentence({"text": "hello", "raw": {}})

        progress_hook(2.0)
        on_sentence({"text": "world", "raw": {"begin_time": 2000, "end_time": 2600}})

    monkeypatch.setattr(asr_client, "transcribe_file", fake_transcribe_file)

    segs = transcribe_wav_to_segments("fake.wav", DashScopeOfflineConfig(api_key="k"))
    assert [s["text"] for s in segs] == ["hello", "world"]
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.0
    assert segs[1]["start"] == 2.0
    assert segs[1]["end"] == 2.6
