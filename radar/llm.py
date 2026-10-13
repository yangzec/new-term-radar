"""Chat models for the radar's two model steps (the name judge, the keyword table).

Providers, chosen by config.json's "llm.provider":
  "xai"   xAI's OpenAI-compatible API (https://api.x.ai/v1), key XAI_API_KEY; the model is
          "llm.model" (a Grok model) whatever judge.model / usecases.model say.
  "aisa"  AIsa's OpenAI-compatible endpoint, key AISA_API_KEY; the model is the step's own.
  "grok-cli"  shells out to the grok-cli binary (logged in with OAuth, no key in the
          environment): `grok-cli chat --json --no-web-search --no-stream --system ...`.
          "llm.cli_model" passes --model (empty: the CLI's default); "llm.cli_timeout" seconds.
  "auto"  (default) xai when XAI_API_KEY is set, else grok-cli when it is installed and
          logged in, else aisa when AISA_API_KEY is set; with none of them the steps skip.
"llm.base_url" points the xai provider at any other OpenAI-compatible server.

LLMError subclasses aisa.AisaError, so every caller's existing error handling still
holds: a failed call never blocks the report.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

from . import UA, aisa

DEFAULTS = {"provider": "auto", "model": "grok-4-1-fast-non-reasoning", "base_url": "https://api.x.ai/v1",
            "key_env": "XAI_API_KEY", "cli": "grok-cli", "cli_model": "", "cli_timeout": 180}
_cfg: dict = dict(DEFAULTS)
TIMEOUT = 90


class LLMError(aisa.AisaError):
    pass


def configure(c: dict | None) -> None:
    global _cfg
    _cfg = {**DEFAULTS, **(c or {})}


_cli_ok: dict[str, bool] = {}


def _cli_path() -> str | None:
    return shutil.which(_cfg["cli"]) or (lambda p: p if os.path.isfile(p) and os.access(p, os.X_OK) else None)(
        os.path.expanduser(f"~/.cargo/bin/{_cfg['cli']}"))


def cli_ready() -> bool:
    """grok-cli is installed and `grok-cli status --json` says logged_in (checked once)."""
    path = _cli_path()
    if not path:
        return False
    if path not in _cli_ok:
        try:
            res = subprocess.run([path, "status", "--json"], capture_output=True, text=True, timeout=30)
            d = json.loads(res.stdout or "{}").get("data") or {}
            _cli_ok[path] = bool(d.get("logged_in")) and not d.get("relogin_required") and not d.get("entitlement_denied")
        except (OSError, ValueError, subprocess.SubprocessError):
            _cli_ok[path] = False
    return _cli_ok[path]


def provider() -> str:
    """xai, grok-cli, aisa, or "" when nothing is usable."""
    p = _cfg["provider"]
    if p != "auto":
        return p
    if os.environ.get(_cfg["key_env"], "").strip():
        return "xai"
    if cli_ready():
        return "grok-cli"
    return "aisa" if aisa.has_key() else ""


def available() -> bool:
    p = provider()
    if p == "grok-cli":
        return cli_ready()
    if p == "xai":
        return bool(os.environ.get(_cfg["key_env"], "").strip())
    return p == "aisa" and aisa.has_key()


JSON_HINT = "\n\nAnswer with the JSON only: no prose, no markdown fences."


def _cli_chat(system: str, user: str) -> str:
    path = _cli_path()
    if not path:
        raise LLMError(f"{_cfg['cli']} not installed")
    t = int(_cfg["cli_timeout"])
    cmd = [path, "chat", "--json", "--no-web-search", "--no-stream", "--timeout", str(t), "--system", system + JSON_HINT]
    if _cfg["cli_model"]:
        cmd += ["--model", _cfg["cli_model"]]
    cmd += ["--prompt", user]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=t + 30)
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"grok-cli timed out after {t}s") from exc
    except OSError as exc:
        raise LLMError(f"grok-cli: {exc}") from exc
    try:
        body = json.loads(res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "{}")
    except ValueError:
        body = {}
    if not body.get("ok"):
        err = body.get("error") or res.stderr or res.stdout
        raise LLMError(f"grok-cli exit {res.returncode}: {' '.join(str(err).split())[:200]}")
    text = (body.get("data") or {}).get("output_text")
    if not isinstance(text, str):
        raise LLMError(f"grok-cli: no output_text in {str(body)[:160]}")
    return text


def chat(model: str, system: str, user: str, max_tokens: int | None = None) -> str:
    p = provider()
    if p == "grok-cli":
        return _cli_chat(system, user)
    if p != "xai" and p != "aisa":
        raise LLMError("no model: set XAI_API_KEY, log in to grok-cli, or set AISA_API_KEY")
    if p == "aisa":
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
