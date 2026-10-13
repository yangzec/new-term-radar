"""Google Trends daily interest, from one of two providers (config.json "trends.provider"):

  "aisa"      DataForSEO Trends explore through AIsa (AISA_API_KEY), $0.0012 a lookup.
  "pytrends"  Google Trends directly through the unofficial pytrends library (pip install
              pytrends). Free, but Google answers 429 quickly, from cloud IPs often on the
              first request; lookups are spaced (trends.delay seconds), retried with
              backoff (trends.retries), and cached per day under state/trends_cache.
  "auto"      (default) aisa when AISA_API_KEY is set, else pytrends.

fetch() takes {call_id: (keywords, date_from, date_to)} and returns
({call_id: [(day, value, value...)]}, cost in micro-USD); a lookup that failed is left out.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

from . import aisa

TOOL = "post_dataforseo_keywords_trends_explore_live"
DEFAULTS = {"provider": "auto", "delay": 8, "retries": 3, "backoff": 60, "geo": "", "cache": True}
_cfg: dict = dict(DEFAULTS)
_cache_dir: Path | None = None
_last = 0.0


def configure(c: dict | None, state: Path | None = None) -> None:
    global _cfg, _cache_dir
    _cfg = {**DEFAULTS, **(c or {})}
    _cache_dir = state / "trends_cache" if state else None


def provider() -> str:
    p = _cfg["provider"]
    if p == "auto":
        return "aisa" if aisa.has_key() else "pytrends"
    return p


def _rows_dfs(task: dict) -> list[tuple]:
    try:
        item = task["result"][0]["items"][0]
    except (KeyError, IndexError, TypeError):
        return []
    rows = []
    for p in item.get("data", []):
        v = [int(x or 0) for x in (p.get("values") or [])]
        rows.append((date.fromisoformat(p["date_from"]), *v))
    return rows


def _aisa(req: dict) -> tuple[dict, int]:
    calls = [{"call_id": cid, "tool": TOOL, "arguments": {"body": [{"keywords": kws, "date_from": f, "date_to": t}]}}
             for cid, (kws, f, t) in req.items()]
    data, cost = aisa.call_all(calls)
    return {cid: _rows_dfs(((data[cid].get("tasks")) or [{}])[0]) for cid in req if cid in data}, cost


def _cache_path(kws: list[str], f: str, t: str) -> Path | None:
    if not (_cache_dir and _cfg["cache"]):
        return None
    h = hashlib.sha1(json.dumps([kws, f, t, _cfg["geo"]]).encode()).hexdigest()[:16]
    return _cache_dir / date.today().isoformat() / f"{h}.json"


def _pytrends_one(client, kws: list[str], f: str, t: str) -> list[tuple]:
    global _last
    cp = _cache_path(kws, f, t)
    if cp and cp.exists():
        return [(date.fromisoformat(r[0]), *r[1:]) for r in json.loads(cp.read_text())]
    err = None
    for attempt in range(_cfg["retries"]):
        wait = _cfg["delay"] - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            _last = time.time()
            client.build_payload(kws, timeframe=f"{f} {t}", geo=_cfg["geo"])
            df = client.interest_over_time()
            break
        except Exception as exc:  # noqa: BLE001  (pytrends raises its own and requests' errors)
            err = exc
            if attempt + 1 < _cfg["retries"]:
                time.sleep(_cfg["backoff"] * (attempt + 1))
    else:
        raise RuntimeError(f"pytrends: {type(err).__name__}: {str(err)[:120]}")
    rows = [] if df is None or df.empty else [
        (idx.date(), *[int(df[k].iloc[i]) if k in df else 0 for k in kws]) for i, idx in enumerate(df.index)]
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps([[d.isoformat(), *v] for d, *v in rows]))
    return rows


def _pytrends(req: dict) -> tuple[dict, int, list[str]]:
    from pytrends.request import TrendReq   # optional dependency
    client = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    out, errors = {}, []
    for cid, (kws, f, t) in req.items():
        try:
            out[cid] = _pytrends_one(client, kws, f, t)
        except RuntimeError as exc:
            errors.append(str(exc))
            if "429" in str(exc) or "TooManyRequests" in str(exc):
                errors.append(f"rate-limited: {len(req) - len(out) - 1} lookups not tried")
                break
    return out, 0, errors


last_errors: list[str] = []


def fetch(req: dict) -> tuple[dict, int]:
    global last_errors
    last_errors = []
    if not req:
        return {}, 0
    if provider() == "aisa":
        return _aisa(req)
    out, cost, last_errors = _pytrends(req)
    return out, cost
