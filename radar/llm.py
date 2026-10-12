"""Chat models for the radar's two model steps (the name judge, the keyword table).

Providers, chosen by config.json's "llm.provider":
  "xai"   xAI's OpenAI-compatible API (https://api.x.ai/v1), key XAI_API_KEY; the model is
          "llm.model" (a Grok model) whatever judge.model / usecases.model say.
  "aisa"  AIsa's OpenAI-compatible endpoint, key AISA_API_KEY; the model is the step's own.
  "auto"  (default) xai when XAI_API_KEY is set, else aisa.
"llm.base_url" points the xai provider at any other OpenAI-compatible server.

LLMError subclasses aisa.AisaError, so every caller's existing error handling still
holds: a failed call never blocks the report.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import UA, aisa

DEFAULTS = {"provider": "auto", "model": "grok-4-1-fast-non-reasoning", "base_url": "https://api.x.ai/v1",
            "key_env": "XAI_API_KEY"}
_cfg: dict = dict(DEFAULTS)
TIMEOUT = 90


class LLMError(aisa.AisaError):
    pass


def configure(c: dict | None) -> None:
    global _cfg
    _cfg = {**DEFAULTS, **(c or {})}


def provider() -> str:
    p = _cfg["provider"]
    if p == "auto":
        return "xai" if os.environ.get(_cfg["key_env"], "").strip() else "aisa"
    return p


def available() -> bool:
    p = provider()
    return bool(os.environ.get(_cfg["key_env"] if p == "xai" else "AISA_API_KEY", "").strip())


def chat(model: str, system: str, user: str, max_tokens: int | None = None) -> str:
    if provider() == "aisa":
        return aisa.chat(model, system, user) if max_tokens is None else aisa.chat(model, system, user, max_tokens)
    max_tokens = max_tokens or 1500
    k = os.environ.get(_cfg["key_env"], "").strip()
    if not k:
        raise LLMError(f"no model key: set {_cfg['key_env']}")
    body = {"model": _cfg["model"], "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(_cfg["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json",
                                                         "Authorization": f"Bearer {k}", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            out = json.load(res)
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}: {' '.join(exc.read().decode('utf-8', 'replace').split())[:200]}") from exc
    except (OSError, ValueError) as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    try:
        return out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected chat answer: {str(out)[:160]}") from exc
