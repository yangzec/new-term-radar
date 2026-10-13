"""External data inbox: files another agent (or you) drops in for the day.

state/inbox/YYYY-MM-DD/   (the directory can be moved with config.json "inbox.dir")
  x_posts.json   {"users":  {"<handle>": <X API v2 GET /2/users/:id/tweets response>},
                  "search": {"<name>":   <X API v2 GET /2/tweets/search/all response>}}
                 Each response: {"data": [{"id", "text", "created_at", "author_id",
                 "public_metrics": {"like_count"}}], "includes": {"users": [{"id", "username"}]}}
  volumes.json   Google Ads Keyword Planner historical metrics:
                 {"metrics": [{"keyword", "avgMonthlySearches",
                               "monthlySearchVolumes": [{"year", "month": "AUGUST", "monthlySearches"}]}]}
                 Keyword Planner leaves out keywords it has no data for; add them with
                 "avgMonthlySearches": 0 so they are not requested again.
  related.json   {"<name>": {"rising": [...], "top": [...]}}  (Trends related queries, Google
                 autocomplete or Keyword Planner ideas; plain lists of query strings)

A missing file or key means "no inbox data": the caller falls back to AIsa when a key is
set, else skips. During `daily` the radar writes requests.json in the same directory: the
names and phrases it wanted and found no data for, so the next run can be supplied.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
                                       "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], 1)}


def folder(cfg: dict, state: Path, today: date | None = None) -> Path:
    base = (cfg.get("inbox") or {}).get("dir") or str(state / "inbox")
    return Path(base) / (today or date.today()).isoformat()


def load(cfg: dict, state: Path, kind: str, today: date | None = None) -> dict | None:
    p = folder(cfg, state, today) / f"{kind}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def tweets(resp: dict) -> list[dict]:
    """X API v2 response -> the radar's tweet rows: text, likes, url, by, created."""
    users = {u.get("id"): u.get("username", "") for u in (resp.get("includes") or {}).get("users") or []}
    out = []
    for tw in resp.get("data") or []:
        by = users.get(tw.get("author_id"), "") or tw.get("username", "")
        tid = tw.get("id", "")
        created = None
        if tw.get("created_at"):
            try:
                created = datetime.fromisoformat(tw["created_at"].replace("Z", "+00:00"))
            except ValueError:
                pass
        out.append({"text": tw.get("text") or "", "likes": int((tw.get("public_metrics") or {}).get("like_count") or 0),
                    "url": tw.get("url") or (f"https://x.com/{by or 'i'}/status/{tid}" if tid else ""),
                    "by": by or "?", "created": created})
    return out


def volumes(data: dict) -> dict[str, dict]:
    """Keyword Planner metrics -> {keyword: {"volume", "month"}} as usecases.volumes() returns."""
    out = {}
    for m in data.get("metrics") or []:
        months = sorted(((int(x["year"]), MONTHS.get(str(x["month"]).upper(), 0) if not str(x["month"]).isdigit()
                          else int(x["month"])) for x in m.get("monthlySearchVolumes") or [] if x.get("year")), reverse=True)
        out[" ".join(str(m.get("keyword", "")).lower().split())] = {
            "volume": int(m.get("avgMonthlySearches") or 0),
            "month": f"{months[0][0]}-{months[0][1]:02d}" if months else ""}
    return out


def supplied(cfg: dict, state: Path, kind: str, today: date | None = None) -> set[str]:
    """What the inbox already holds for a request kind (lowercased), so requests.json never
    asks again for something supplied."""
    if kind in ("x_users", "x_search"):
        part = (load(cfg, state, "x_posts", today) or {}).get("users" if kind == "x_users" else "search") or {}
        return {" ".join(k.lower().lstrip("@").split()) for k in part}
    if kind == "volumes":
        return set(volumes(load(cfg, state, "volumes", today) or {}))
    if kind == "related":
        return {" ".join(k.lower().split()) for k in load(cfg, state, "related", today) or {}}
    return set()


def request(cfg: dict, state: Path, kind: str, items: list[str], today: date | None = None) -> None:
    """Note what the radar wanted and had no data for (requests.json, merged)."""
    d = folder(cfg, state, today)
    p = d / "requests.json"
    if not items and not p.exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    have = supplied(cfg, state, kind, today)
    cur[kind] = sorted(i for i in set(cur.get(kind, [])) | set(items) if " ".join(i.lower().lstrip("@").split()) not in have)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
