from typing import Any, Dict

from tools.language_utils import normalize_language


def _pick_secret(loaded: Dict[str, Any], current: Dict[str, Any], key: str) -> str:
    loaded_val = str(loaded.get(key) or "").strip()
    if loaded_val:
        return loaded_val
    current_val = str(current.get(key) or "").strip()
    if current_val:
        return current_val
    return ""


def merge_redacted_model_cfg(loaded: Any, current: Any) -> Dict[str, Any]:
    """Merge redacted model config with in-memory secrets."""
    if not isinstance(loaded, dict):
        loaded = {}
    if not isinstance(current, dict):
        current = {}
    out: Dict[str, Any] = dict(loaded)

    loaded_llm = loaded.get("llm") if isinstance(loaded.get("llm"), dict) else {}
    current_llm = current.get("llm") if isinstance(current.get("llm"), dict) else {}
    if loaded_llm or current_llm:
        merged_llm = dict(loaded_llm)
        llm_key = _pick_secret(loaded_llm, current_llm, "api_key")
        if llm_key:
            merged_llm["api_key"] = llm_key
        out["llm"] = merged_llm

    loaded_asr = loaded.get("asr") if isinstance(loaded.get("asr"), dict) else {}
    current_asr = current.get("asr") if isinstance(current.get("asr"), dict) else {}
    if loaded_asr or current_asr:
        merged_asr = dict(loaded_asr)
        asr_key = _pick_secret(loaded_asr, current_asr, "api_key")
        if asr_key:
            merged_asr["api_key"] = asr_key
        out["asr"] = merged_asr

    return out


def merge_session_meta(cfg: Any, meta: Any) -> Dict[str, Any]:
    """Apply non-sensitive session metadata (mode/language) onto a model config."""
    if not isinstance(cfg, dict):
        cfg = {}
    if not isinstance(meta, dict):
        return dict(cfg)
    out: Dict[str, Any] = dict(cfg)
    mode = str(meta.get("mode") or "").strip()
    if mode in ("online", "offline"):
        out["mode"] = mode
    if isinstance(out.get("llm"), dict) and meta.get("llm_language") is not None:
        out["llm"]["language"] = normalize_language(
            meta.get("llm_language"),
            default=str(out["llm"].get("language") or "zh"),
            allow_auto=True,
        )
    if isinstance(out.get("asr"), dict) and meta.get("asr_language") is not None:
        out["asr"]["language"] = normalize_language(
            meta.get("asr_language"),
            default=str(out["asr"].get("language") or "auto"),
            allow_auto=True,
        )
    return out
