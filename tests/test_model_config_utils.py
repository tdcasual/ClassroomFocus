from tools.model_config_utils import merge_redacted_model_cfg, merge_session_meta


def test_merge_redacted_model_cfg_merges_missing_keys():
    loaded = {"llm": {"api_key": ""}, "asr": {"api_key": ""}}
    current = {"llm": {"api_key": "llm-key"}, "asr": {"api_key": "asr-key"}}
    merged = merge_redacted_model_cfg(loaded, current)
    assert merged["llm"]["api_key"] == "llm-key"
    assert merged["asr"]["api_key"] == "asr-key"


def test_merge_redacted_model_cfg_preserves_existing_keys():
    loaded = {"llm": {"api_key": "keep-llm"}, "asr": {"api_key": "keep-asr"}}
    current = {"llm": {"api_key": "new-llm"}, "asr": {"api_key": "new-asr"}}
    merged = merge_redacted_model_cfg(loaded, current)
    assert merged["llm"]["api_key"] == "keep-llm"
    assert merged["asr"]["api_key"] == "keep-asr"


def test_merge_session_meta_applies_language_and_mode():
    cfg = {"mode": "offline", "llm": {"language": "zh"}, "asr": {"language": "auto"}}
    meta = {"mode": "online", "llm_language": "en", "asr_language": "zh"}
    merged = merge_session_meta(cfg, meta)
    assert merged["mode"] == "online"
    assert merged["llm"]["language"] == "en"
    assert merged["asr"]["language"] == "zh"
