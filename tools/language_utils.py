from typing import Optional


def normalize_language(value: Optional[str], default: str = "zh", allow_auto: bool = False) -> str:
    """Normalize language hints to 'zh', 'en', or 'auto'."""
    if value is None:
        return default
    v = str(value).strip().lower()
    if not v:
        return default
    if v in ("auto", "detect", "default"):
        return "auto" if allow_auto else default
    if v in ("zh", "zh-cn", "zh-hans", "zh-hans-cn", "zh_cn", "cn", "chinese", "中文"):
        return "zh"
    if v in ("en", "en-us", "en-gb", "en_us", "en_gb", "english", "英文"):
        return "en"
    return default


def llm_language_hint(lang: str) -> str:
    """Return a short Chinese hint for LLM output language."""
    if lang == "en":
        return "输出英文；如输入非英文，请翻译为英文。"
    if lang == "zh":
        return "输出中文；如输入非中文，请翻译为中文。"
    return "根据原文语言输出。"


def asr_language_param(lang: str) -> Optional[str]:
    """Return ASR language param or None for auto-detect."""
    if not lang or lang == "auto":
        return None
    return lang


def resolve_asr_language(asr_language: Optional[str], llm_language: Optional[str]) -> str:
    """Resolve ASR language with LLM fallback when ASR is auto."""
    asr_lang = normalize_language(asr_language, default="auto", allow_auto=True)
    llm_lang = normalize_language(llm_language, default="zh", allow_auto=True)
    if asr_lang == "auto" and llm_lang in ("zh", "en"):
        return llm_lang
    return asr_lang
