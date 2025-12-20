from tools.language_utils import normalize_language, llm_language_hint, asr_language_param, resolve_asr_language


def test_normalize_language_zh_aliases():
    assert normalize_language("zh") == "zh"
    assert normalize_language("ZH-cn") == "zh"
    assert normalize_language("Chinese") == "zh"
    assert normalize_language("cn") == "zh"


def test_normalize_language_en_aliases():
    assert normalize_language("en") == "en"
    assert normalize_language("EN-us") == "en"
    assert normalize_language("english") == "en"


def test_normalize_language_unknown_falls_back():
    assert normalize_language("jp", default="zh") == "zh"
    assert normalize_language("jp", default="en") == "en"


def test_normalize_language_auto():
    assert normalize_language("auto", default="zh", allow_auto=True) == "auto"
    assert normalize_language("auto", default="zh", allow_auto=False) == "zh"


def test_llm_language_hint():
    assert llm_language_hint("zh") == "输出中文；如输入非中文，请翻译为中文。"
    assert llm_language_hint("en") == "输出英文；如输入非英文，请翻译为英文。"
    assert llm_language_hint("auto") == "根据原文语言输出。"


def test_asr_language_param():
    assert asr_language_param("auto") is None
    assert asr_language_param("") is None
    assert asr_language_param("zh") == "zh"


def test_resolve_asr_language():
    assert resolve_asr_language("auto", "zh") == "zh"
    assert resolve_asr_language(None, "en") == "en"
    assert resolve_asr_language("en", "zh") == "en"
