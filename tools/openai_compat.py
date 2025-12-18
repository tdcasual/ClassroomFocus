import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


def _normalize_v1_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        base_url = "https://api.openai.com"
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}".strip())
    return "\n\n".join(parts).strip() + "\n"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_json_array(text: str) -> Optional[List[Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        arr = json.loads(text)
        return arr if isinstance(arr, list) else None
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def _extract_text_from_responses(resp: Dict[str, Any]) -> str:
    # OpenAI Responses API commonly returns: { "output": [ { "content": [ {"type":"output_text","text":"..."} ] } ] }
    # Some implementations also return { "output_text": "..." }.
    if not isinstance(resp, dict):
        return ""
    if isinstance(resp.get("output_text"), str):
        return resp.get("output_text") or ""
    out = resp.get("output")
    if isinstance(out, list):
        texts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        texts.append(c["text"])
            # Some providers put text directly on the output item.
            if isinstance(item.get("text"), str):
                texts.append(item.get("text"))
        return "\n".join(t for t in texts if t).strip()
    return ""


def _extract_text_from_chat_completions(resp: Dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    # fallback for providers that return {choices:[{text:"..."}]}
    if isinstance(choices[0], dict) and isinstance(choices[0].get("text"), str):
        return choices[0]["text"]
    return ""


@dataclass
class OpenAICompatConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: float = 60.0


class OpenAICompat:
    """Minimal OpenAI-compatible client using `requests`.

    Compatibility strategy:
    - Prefer `/v1/responses` (commonly needed by GPT-5+ style models).
    - Fall back to `/v1/chat/completions` for older/partial implementations.
    - Handle token parameter name differences (`max_output_tokens` vs `max_tokens`).
    """

    def __init__(self, cfg: OpenAICompatConfig):
        self.cfg = cfg
        self.v1 = _normalize_v1_base_url(cfg.base_url)

    @classmethod
    def from_env(cls, model_env: str = "OPENAI_MODEL") -> Optional["OpenAICompat"]:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
        if not api_key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_ENDPOINT") or "https://api.openai.com"
        model = os.getenv(model_env) or "gpt-4o-mini"
        timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
        return cls(OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model, timeout_sec=timeout))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.cfg.timeout_sec)
        try:
            j = r.json()
        except Exception:
            j = {"error": {"message": r.text}}
        return r.status_code, j

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        model = model or self.cfg.model
        prompt = _messages_to_prompt(messages)

        # 1) Try Responses API (GPT-5+ often prefers this).
        responses_url = f"{self.v1}/responses"
        payload = {
            "model": model,
            "input": prompt,
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        }
        code, j = self._post_json(responses_url, payload)
        if code >= 200 and code < 300:
            txt = _extract_text_from_responses(j)
            if txt:
                return txt

        # Retry with `max_tokens` (some compatible backends reuse old param name).
        payload2 = dict(payload)
        payload2.pop("max_output_tokens", None)
        payload2["max_tokens"] = int(max_tokens)
        code2, j2 = self._post_json(responses_url, payload2)
        if code2 >= 200 and code2 < 300:
            txt = _extract_text_from_responses(j2)
            if txt:
                return txt
        # Retry with `max_completion_tokens` (some newer backends use this name).
        payload3 = dict(payload)
        payload3.pop("max_output_tokens", None)
        payload3["max_completion_tokens"] = int(max_tokens)
        code3, j3 = self._post_json(responses_url, payload3)
        if code3 >= 200 and code3 < 300:
            txt = _extract_text_from_responses(j3)
            if txt:
                return txt

        # 2) Fall back to Chat Completions.
        chat_url = f"{self.v1}/chat/completions"
        chat_payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        code4, j4 = self._post_json(chat_url, chat_payload)
        if code4 >= 200 and code4 < 300:
            txt = _extract_text_from_chat_completions(j4)
            if txt:
                return txt
        # Retry with `max_completion_tokens` for newer server implementations.
        chat_payload2 = dict(chat_payload)
        chat_payload2.pop("max_tokens", None)
        chat_payload2["max_completion_tokens"] = int(max_tokens)
        code5, j5 = self._post_json(chat_url, chat_payload2)
        if code5 >= 200 and code5 < 300:
            txt = _extract_text_from_chat_completions(j5)
            if txt:
                return txt

        # Raise a helpful error with best available message.
        err = None
        for obj in (j, j2, j3, j4, j5):
            msg = None
            if isinstance(obj, dict):
                e = obj.get("error")
                if isinstance(e, dict) and isinstance(e.get("message"), str):
                    msg = e.get("message")
                elif isinstance(obj.get("message"), str):
                    msg = obj.get("message")
            if msg:
                err = msg
                break
        raise RuntimeError(err or f"OpenAI-compatible request failed (HTTP {code}/{code2}/{code3}/{code4}/{code5})")

    def generate_json_object(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        txt = self.generate_text(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature)
        obj = _extract_json_object(txt)
        if obj is None:
            raise ValueError("Model did not return a JSON object.")
        return obj

    def generate_json_array(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.2,
    ) -> List[Any]:
        txt = self.generate_text(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature)
        arr = _extract_json_array(txt)
        if arr is None:
            raise ValueError("Model did not return a JSON array.")
        return arr

    def transcribe_audio(
        self,
        audio_path: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "verbose_json",
    ) -> Dict[str, Any]:
        """Call OpenAI-compatible `/v1/audio/transcriptions`."""
        model = model or os.getenv("OPENAI_ASR_MODEL") or "whisper-1"
        url = f"{self.v1}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        data = {"model": model, "response_format": response_format}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        # request segment timestamps when supported
        data["timestamp_granularities[]"] = "segment"
        with open(audio_path, "rb") as fh:
            files = {"file": (os.path.basename(audio_path), fh, "audio/wav")}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=self.cfg.timeout_sec)
        try:
            j = r.json()
        except Exception:
            j = {"error": {"message": r.text}}
        if r.status_code < 200 or r.status_code >= 300:
            msg = None
            e = j.get("error") if isinstance(j, dict) else None
            if isinstance(e, dict) and isinstance(e.get("message"), str):
                msg = e.get("message")
            raise RuntimeError(msg or f"audio transcription failed (HTTP {r.status_code})")
        return j
