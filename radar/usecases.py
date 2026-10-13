"""The generic terms behind the names: what people do with a new thing, as search terms.

A name ("SynthID Detector") has a short life as a keyword and belongs to someone. The
job it does ("ai image detector") is what people keep searching, and nobody owns it.
For the day's new names this step:

  1. collects evidence: popular posts that mention the name (AIsa, X advanced search)
     and, for a repository, its README;
  2. asks a model which jobs the evidence shows people doing, and the keyword an
     ordinary person would type for each job. A phrase built on another product's name
     ("photoshop alternative") is kept and marked `brand`: demand often hangs on the
     incumbent, and that is yours to weigh;
  2b. with a GoAnyAPI key (optional), also takes Google Trends' related queries for the
     name: what people really searched together with it. Queries that hold the name are
     its long-tails ("photocraft download"); the others join the phrases; those that are
     themselves names of other new things ("gpt image 2.5" beside "nano banana 2.1") are
     handed back as leads of kind "cosearch";
  3. screens them: a second call scores how well the name answers each phrase, and the
     loosely related ones are dropped (a keyword taken from a passing comparison);
  4. looks the rest up: Google Ads monthly searches (one batch call);
  5. reads Google Trends for the names and for the phrases with enough volume, twice
     each: alone, for the shape of the curve (when it started rising, its peak, what it
     is doing now), and next to a benchmark keyword ("gpts" by default), for the size.
     A generic term's searches a day come from its own Google Ads volume, moved to today
     along its own curve. The benchmark is sized from those terms, and a name, which has
     no volume data yet, is sized against the benchmark.

The result is one table, names and generic terms together, each with a decision
(table()). Phrases with measurable volume are also kept as leads of kind "usecase" and
go through the same 14-day review as the names.

What it cannot do: on the day a model is announced the posts are announcements, and
no job shows yet; each name is therefore read twice, the day it appears and
`recheck_days` later. Google Ads volumes run about two months behind, so they tell how
big a need is, not whether this release moved it; Trends tells that, but reads nothing
for terms under a few hundred searches a month. Most proposed phrases have no volume.

Costs seen in October 2026: $0.003 per name for the posts, $0.09 for the volume batch
(up to 1,000 phrases), $0.0012 per Trends lookup (two per row of the table), plus the model's tokens.
"""
from __future__ import annotations

import base64
import json
import os
import re
import statistics
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from . import UA, aisa, goanyapi, inbox, llm, review, snap, trends

POSTS_KEPT = 12
POST_CHARS = 400
README_CHARS = 3000
MIN_FAVES = 50
SEARCH_DAYS = 14
TRENDS_DAYS, RECENT_DAYS = 90, 7
RISE_WINDOW = 30           # a rise counts when it happened in the last 30 days
RISE_DAYS = 2              # and lasted at least two days
MOMENTUM_MIN_BASE = 5      # of 100: the week before must reach this for a week-on-week ratio
NOW_DAYS = 3               # a peak in the last three settled days is still rising
HOLD_SHARE = 0.5           # at half its peak or more in those days, a term is holding
CALIBRATE_MIN = 1000       # monthly searches a term needs to help size the benchmark
NEW_BASE_MAX = 10          # a baseline under 10 (of 100) means the term barely existed before
MAX_PER_NAME = 6
STATE_DAYS = 30
SEARCH_TOOL = "get_twitter_tweet_advanced_search"
VOLUME_TOOL = "post_dataforseo_keywords_gads_search_volume_live"
SYSTEM = (
    "You help an SEO researcher who builds tool pages around new AI releases. You get the NAME of a new model, product "
    "or repository and EVIDENCE: popular posts about it and, for a repository, its README.\n"
    "Step 1: work out the concrete jobs people are doing with it, according to the evidence.\n"
    "Step 2: for each job, write the KEYWORD a keyword tool would list for it: the short head term people type into "
    "Google when they want that job done and have never heard of this product.\n"
    "A keyword is a noun phrase for a tool or a task, 2 to 4 words, lowercase: 'ai image detector', 'psd editor online', "
    "'run llm locally', 'photo to anime', 'ai dance video generator', 'remove background from video'. Not a sentence, "
    "not a feature description ('generate ultra detailed images', 'combine multiple reference images'), not a quality "
    "('fast image generation'), not news.\n"
    "Never use the NAME itself or the name of its maker. Prefer keywords with no names at all. When the evidence shows "
    "people comparing it to an established product or using it on a named platform, you may add up to two keywords "
    "built on that established name ('photoshop alternative', 'debloat macos') and set `brand` to true for them. Set "
    "`brand` to true for any keyword that contains a brand, product, company, operating system, person, studio, "
    "franchise or character name; false otherwise.\n"
    "Every keyword needs a short quote from the evidence showing people doing that job. No evidence of a concrete job "
    "(only an announcement, prices or benchmarks): return an empty list. Do not guess.\n"
    "Answer with JSON only: {\"what_it_is\": \"one sentence\", \"use_cases\": [{\"phrase\": \"...\", \"evidence\": "
    "\"short quote\", \"brand\": false}]}, at most 6, the most specific job first."
)


def due(names: list[str], seen: dict[str, dict], today: date, per_day: int, recheck_days: int) -> list[str]:
    """Names to read today: new ones first, then those read once `recheck_days` ago or earlier."""
    new = [n for n in names if n not in seen]
    cutoff = (today - timedelta(days=recheck_days)).isoformat()
    again = sorted((n for n, s in seen.items() if s["runs"] == 1 and s["first"] <= cutoff), key=lambda n: seen[n]["first"])
    return list(dict.fromkeys(new + again))[:per_day]


def _readme(repo: str) -> str:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": UA}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{repo}/readme", headers=headers), timeout=30) as res:
            return base64.b64decode(json.load(res)["content"]).decode("utf-8", "replace")[:README_CHARS]
    except (OSError, KeyError, ValueError):
        return ""


def evidence(names: list[str], repos: dict[str, str], today: date, supplied: dict | None = None) -> tuple[dict[str, dict], int]:
    """{name: {"posts": [...], "readme": str}} and the cost in micro-USD. `supplied`: the
    inbox's {name: X API v2 search response}; names not in it go to AIsa when a key is set."""
    supplied = supplied or {}
    since = (today - timedelta(days=SEARCH_DAYS)).isoformat()
    ask = [n for n in names if n not in supplied] if os.environ.get("AISA_API_KEY", "").strip() else []
    calls = [{"call_id": f"n{names.index(n)}", "tool": SEARCH_TOOL, "arguments": {
        "query": f'"{n}" min_faves:{MIN_FAVES} -filter:replies since:{since}', "queryType": "Top"}} for n in ask]
    data, cost = aisa.call_all(calls) if calls else ({}, 0)
    for i, n in enumerate(names):
        if n in supplied:
            data[f"n{i}"] = {"_inbox": sorted(inbox.tweets(supplied[n]), key=lambda p: -p["likes"])}
    out = {}
    for i, n in enumerate(names):
        d = data.get(f"n{i}") or {}
        if "_inbox" in d:
            posts = [{k: p[k] for k in ("text", "likes", "by", "url")} for p in d["_inbox"]][:POSTS_KEPT]
            out[n] = {"posts": posts, "readme": _readme(repos[n]) if repos.get(n) else "", "repo": repos.get(n, "")}
            continue
        tweets = d.get("tweets") or (d.get("data") or {}).get("tweets") or []
        posts = sorted(({"text": " ".join((t.get("text") or "").split()), "likes": int(t.get("likeCount") or 0),
                         "by": (t.get("author") or {}).get("userName") or "?", "url": t.get("url", "")} for t in tweets),
                       key=lambda p: -p["likes"])[:POSTS_KEPT]
        out[n] = {"posts": posts, "readme": _readme(repos[n]) if repos.get(n) else "", "repo": repos.get(n, "")}
    return out, cost


def parse(answer: str, name: str) -> list[dict]:
    """The model's use cases, cleaned: lowercased, 2 to 5 words, never the name itself, no repeats."""
    m = re.search(r"\{.*\}", answer, re.S)
    try:
        items = json.loads(m.group(0)).get("use_cases", []) if m else []
    except (ValueError, AttributeError):
        return []
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        phrase = " ".join(str(it.get("phrase", "")).lower().split()) if isinstance(it, dict) else ""
        if not 2 <= len(phrase.split()) <= 5 or name in phrase or phrase in seen:
            continue
        seen.add(phrase)
        out.append({"phrase": phrase, "evidence": " ".join(str(it.get("evidence", "")).split())[:160], "brand": bool(it.get("brand"))})
    return out[:MAX_PER_NAME]


def propose(model: str, name: str, ev: dict) -> list[dict]:
    if not ev["posts"] and not ev["readme"]:
        return []
    posts = "\n".join(f"- ({p['likes']} likes) @{p['by']}: {p['text'][:POST_CHARS]}" for p in ev["posts"])
    user = f"NAME: {name}\n\nPOSTS:\n{posts or '(none found)'}\n" + (f"\nREADME of {ev['repo']}:\n{ev['readme']}\n" if ev["readme"] else "")
    try:
        return parse(llm.chat(model, SYSTEM, user, max_tokens=900), name)
    except aisa.AisaError:
        return []


SCREEN = (
    "You check keyword candidates for an SEO researcher. NAME is a new model, product or repository; EVIDENCE is what "
    "people posted about it. For each candidate keyword give three things.\n"
    "`score`, 0.0 to 1.0: how well NAME answers that search. Would someone who typed the keyword be served by a page "
    "about using NAME for it? High: the keyword names the main job NAME does, as the evidence shows it ('ai image "
    "detector' for a tool that tells whether an image is AI-made; 'photoshop alternative' for a free clone of that "
    "editor). Low: a side mention, a comparison someone made in passing ('chatgpt alternative' because one post "
    "compared two models), a broad category NAME is only one of thousands in ('ai coding assistant' for a general "
    "language model), or a job the evidence does not show.\n"
    "`brand`, true or false: the keyword contains a brand, product, company, operating system, person, studio, "
    "franchise or character name ('photoshop alternative', 'free up mac disk space').\n"
    "`name`, true or false: the keyword is itself the name of another specific product, model or tool, with or "
    "without its maker ('gpt image 2.5', 'gemini 4 argon', 'mistral large 4'), not a task or a kind of tool.\n"
    "Answer with JSON only, one object per candidate, in the same order: "
    "[{\"i\": 0, \"score\": 0.9, \"brand\": false, \"name\": false}, ...]"
)
RELEVANT_MIN = 0.6
RELATED_KEPT = 12          # related queries per name that go to the screen
COSEARCH_KEPT = 5          # other names kept as leads per name


def split_related(name: str, related: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    """Related queries sorted into the name's own long-tails ("photocraft download") and
    candidate phrases. One-word queries ("github", "google") say nothing and are dropped."""
    words = name.split()
    tails, cands, seen = [], [], set()
    for kind in ("rising", "top"):
        for q in related.get(kind, []):
            if q in seen or q == name:
                continue
            seen.add(q)
            if all(w in q.split() for w in words) or name in q:
                tails.append(q)
            elif len(q.split()) >= 2:
                cands.append({"phrase": q, "brand": False, "source": "searched", "rising": kind == "rising",
                              "evidence": f"searched on Google together with {name} ({kind})"})
    return tails, cands[:RELATED_KEPT]


def screen(model: str, name: str, ev: dict, cands: list[dict]) -> list[dict]:
    """The candidates with the model's view of each: `relevance`, `brand` (added to what
    the proposer said), `is_name`. A failed call returns them as they came, unscored: the
    screen narrows the table, it never empties it."""
    if not cands:
        return []
    posts = "\n".join(f"- ({p['likes']} likes) @{p['by']}: {p['text'][:POST_CHARS]}" for p in ev["posts"][:6])
    user = f"NAME: {name}\n\nEVIDENCE:\n{posts or ev['readme'][:1200]}\n\nCANDIDATES:\n" + json.dumps(
        [{"i": i, "keyword": c["phrase"]} for i, c in enumerate(cands)], ensure_ascii=False)
    try:
        answer = llm.chat(model, SCREEN, user, max_tokens=600)
    except aisa.AisaError:
        return cands
    m = re.search(r"\[.*\]", answer, re.S)
    try:
        views = {int(x["i"]): x for x in json.loads(m.group(0))} if m else {}
    except (ValueError, KeyError, TypeError):
        return cands
    out = []
    for i, c in enumerate(cands):
        v = views.get(i)
        try:
            out.append(dict(c, relevance=float(v["score"]), brand=c["brand"] or bool(v.get("brand")), is_name=bool(v.get("name")))
                       if v else c)
        except (KeyError, TypeError, ValueError):
            out.append(c)
    return out


def other_names(name: str, viewed: list[dict]) -> list[dict]:
    """Names of other things that are rising in searches beside this one: leads of their
    own. A query sharing a word of the name is the same family ("google nano banana",
    "nano banana pro" beside "nano banana 2.1"), not another thing."""
    family = {w for w in name.split() if w.isalpha() and len(w) > 2}
    out = [{"term": u["phrase"], "with": name} for u in viewed
           if u.get("is_name") and u.get("rising") and not family & set(u["phrase"].split())]
    return out[:COSEARCH_KEPT]


def relevant(cands: list[dict]) -> list[dict]:
    """Candidates that pass the screen: not another thing's name, and related enough."""
    return [c for c in cands if not c.get("is_name") and c.get("relevance", 1.0) >= RELEVANT_MIN]


def volumes(phrases: list[str], location_code: int, supplied: dict | None = None) -> tuple[dict[str, dict], int]:
    """{phrase: {"volume", "month"}} from Google Ads: the inbox's volumes.json first (already
    parsed by inbox.volumes), the rest in one AIsa batch call when a key is set."""
    supplied = supplied or {}
    have = {p: supplied[p] for p in phrases if p in supplied}
    phrases = [p for p in phrases if p not in supplied]
    if not phrases or not os.environ.get("AISA_API_KEY", "").strip():
        return have, 0
    got, cost = _aisa_volumes(phrases, location_code)
    return {**got, **have}, cost


def _aisa_volumes(phrases: list[str], location_code: int) -> tuple[dict[str, dict], int]:
    data, cost = aisa.call([{"call_id": "vol", "tool": VOLUME_TOOL, "arguments": {"body": [{
        "keywords": phrases, "location_code": location_code, "language_code": "en"}]}}])
    out = {}
    for r in ((data.get("vol") or {}).get("tasks") or [{}])[0].get("result") or []:
        ms = r.get("monthly_searches") or []
        out[r["keyword"]] = {"volume": int(r.get("search_volume") or 0),
                             "month": f"{ms[0]['year']}-{ms[0]['month']:02d}" if ms else ""}
    return out, cost


def shape(points: list[tuple[date, int]], today: date) -> dict:
    """What a term's own Trends curve says: is it new, when did it start rising, where is
    the peak, what is it doing now. `points` is the daily series on its own 0-100 scale;
    its unsettled last days are dropped, and "now" means the last settled days. A rise is
    judged inside the last RISE_WINDOW days against the level before them, so an old peak
    does not count, and it needs RISE_DAYS days above the line: a term with few searches
    shows single-day spikes out of nothing ("debloat macos": 0, 0, 100, 0, 0)."""
    points = review.settled(points, today)
    top = max((v for _, v in points), default=0)
    if not top:
        return {"no_data": True}
    # Back onto 0-100: when a dropped day held the series' 100, the real days sit squeezed
    # under 10 and no threshold in points would mean anything.
    points = [(d, round(v * 100 / top)) for d, v in points]
    end = points[-1][0]
    cut = end - timedelta(days=RISE_WINDOW)
    old = [v for d, v in points if d < cut]
    window = [(d, v) for d, v in points if d >= cut]
    base = statistics.mean(old) if old else 0.0
    peak_day, peak = max(window, key=lambda p: (p[1], p[0]))
    line = max(base * review.RISE_FACTOR, base + review.RISE_MIN)
    above = [d for d, v in window if v >= line]
    rose = len(above) >= RISE_DAYS
    week, before = [v for _, v in points[-RECENT_DAYS:]], [v for _, v in points[-2 * RECENT_DAYS:-RECENT_DAYS]]
    out = {"no_data": False, "base": round(base, 1), "peak": peak, "peak_day": peak_day.isoformat(),
           "rose": rose, "new": rose and base < NEW_BASE_MAX,
           # the last settled week against the week before it; None when that week held too
           # little to divide by ("ai audio detector" read x43 on a week of near-zeros)
           "momentum": round(statistics.mean(week) / statistics.mean(before), 1)
           if before and statistics.mean(before) >= MOMENTUM_MIN_BASE else None}
    if not rose:
        return {**out, "now": "flat"}
    out["started"] = above[0].isoformat()
    latest = max(v for d, v in window if d > end - timedelta(days=NOW_DAYS))
    out["now"] = "rising" if (end - peak_day).days < NOW_DAYS else "holding" if latest >= HOLD_SHARE * peak else "fading"
    return out


def _mean(values: list[int]) -> float:
    return statistics.mean(values) if values else 0.0


def _in_month(day: date, month: str) -> bool:
    return day.isoformat().startswith(month)


def own_per_day(points: list[tuple[date, int]], volume: int, month: str, today: date) -> int | None:
    """Searches a day now, from the term's own numbers: its Google Ads volume for `month`
    scaled by how its Trends level in the last RECENT_DAYS compares with that month."""
    points = review.settled(points, today)
    anchor = _mean([v for d, v in points if _in_month(d, month)])
    recent = _mean([v for _, v in points[-RECENT_DAYS:]])
    return round(volume / 30 * recent / anchor) if anchor and volume else None


def bench_daily(rows: list[dict]) -> float | None:
    """The benchmark's searches a day, worked back from the terms whose volume is known:
    in the month of the volume data, a term at a third of the benchmark's level with
    3,000 searches a day puts the benchmark at 9,000. The median over the terms with
    at least CALIBRATE_MIN searches a month; Trends is too coarse below that."""
    found = []
    for r in rows:
        pair, month = r.get("_pair") or [], r.get("month")
        if not month or r.get("volume", 0) < CALIBRATE_MIN:
            continue
        term = _mean([v for d, v, _ in pair if _in_month(d, month)])
        bench = _mean([b for d, _, b in pair if _in_month(d, month)])
        if term and bench:
            found.append(r["volume"] / 30 * bench / term)
    return statistics.median(found) if found else None


def size(pair: list[tuple[date, int, int]], today: date, daily: float | None) -> dict:
    """A term next to the benchmark on one scale: its peak against the benchmark's usual
    level, and, when the benchmark's size is known, searches a day now."""
    pair = review.settled(pair, today)
    level = _mean([b for _, _, b in pair])
    if not level:
        return {}
    window = pair[-RISE_WINDOW:]
    out = {"peak_vs": max(v for _, v, _ in window), "bench_level": round(level, 1)}
    if daily:
        out["per_day"] = round(_mean([v for _, v, _ in pair[-RECENT_DAYS:]]) / level * daily)
    return out


def decide(s: dict, per_day: int | None, min_daily: int) -> str:
    """prioritize: rose, still up, and big enough. watch: rose but small; too early to
    read; or a term that did not move but has steady demand of that size (a need the
    new thing serves, not one it created). skip: fading, or flat and small."""
    big = (per_day or 0) >= min_daily
    if s.get("no_data"):
        return "watch"
    if s["now"] == "fading":
        return "skip"
    if s["now"] == "flat":
        return "watch" if big else "skip"
    return "prioritize" if big else "watch"


def _pairs(task: dict) -> list[tuple[date, int, int]]:
    try:
        item = task["result"][0]["items"][0]
    except (KeyError, IndexError, TypeError):
        return []
    rows = []
    for p in item.get("data", []):
        v = (p.get("values") or []) + [0, 0]
        rows.append((date.fromisoformat(p["date_from"]), int(v[0] or 0), int(v[1] or 0)))
    return rows


def fetch_series(keywords: list[str], benchmark: str, today: date) -> tuple[dict[str, dict], int]:
    """{keyword: {"solo": [(day, value)], "pair": [(day, value, benchmark value)]}}: two Trends lookups each."""
    f, t = (today - timedelta(days=TRENDS_DAYS)).isoformat(), today.isoformat()
    req = {}
    for i, k in enumerate(keywords):
        req[f"s{i}"] = ([k], f, t)
        req[f"p{i}"] = ([k, benchmark], f, t)
    data, cost = trends.fetch(req)
    solo = lambda cid: [(d, (v + [0])[0]) for d, *v in data.get(cid, [])]  # noqa: E731
    pair = lambda cid: [(d, *(v + [0, 0])[:2]) for d, *v in data.get(cid, [])]  # noqa: E731
    return {k: {"solo": solo(f"s{i}"), "pair": pair(f"p{i}")} for i, k in enumerate(keywords)}, cost


def measure(rows: list[dict], series: dict[str, dict], min_daily: int, today: date) -> float | None:
    """Fills each row with its shape, size and decision. Returns the benchmark's calibrated size."""
    for r in rows:
        r["_pair"] = series.get(r["phrase"], {}).get("pair") or []
    daily = bench_daily(rows)
    for r in rows:
        solo = series.get(r["phrase"], {}).get("solo") or []
        s, z = shape(solo, today), size(r.pop("_pair"), today, daily)
        own = own_per_day(solo, r.get("volume", 0), r.get("month", ""), today)
        if own is not None:
            z["per_day"] = own      # a term's own volume is a better anchor than the benchmark
        r.update({**s, **z, "decision": decide(s, z.get("per_day"), min_daily)})
    return daily


def run(cfg: dict, state: Path, names: list[str], repos: dict[str, str], today: date | None = None) -> dict:
    """names: today's named leads, most important first. repos: {name: "owner/repo"} where known."""
    c = cfg["usecases"]
    today = today or date.today()
    path = state / "usecases.json.gz"
    seen: dict[str, dict] = snap.load(path) or {}
    box_x = inbox.load(cfg, state, "x_posts", today) or {}
    box_rel = inbox.load(cfg, state, "related", today) or {}
    # Names an earlier run asked about (requests.json) and the inbox now answers for come
    # first: the day's leads move between runs, the supplied evidence should not go unused.
    # A name read earlier without any evidence ("no_ev") is read again once evidence arrives.
    box_x["search"] = {" ".join(n.lower().split()): v for n, v in (box_x.get("search") or {}).items()}
    answered = list(box_x["search"])
    names = list(dict.fromkeys(answered + list(names)))
    retry = [n for n in answered if (seen.get(n) or {}).get("no_ev")]
    todo = list(dict.fromkeys(retry + due(names, seen, today, c["per_day"], c["recheck_days"])))[:c["per_day"]]
    if not todo:
        return {"names": [], "rows": [], "proposed": 0, "cost_micros": 0}
    box_vol = inbox.volumes(inbox.load(cfg, state, "volumes", today) or {})
    ev, cost = evidence(todo, repos, today, box_x.get("search"))
    inbox.request(cfg, state, "x_search", [n for n in todo if n not in (box_x.get("search") or {})], today)
    proposed: dict[str, dict] = {}
    asked, c0 = 0, 0
    tails: dict[str, list[str]] = {}
    cosearch: list[dict] = []
    related_errors: list[str] = []
    use_related = (bool(goanyapi.key()) or bool(box_rel)) and c.get("related", True)
    for name in todo:
        cands = [dict(u, source="model") for u in propose(c["model"], name, ev[name])]
        if use_related:
            related, cost_r = {}, 0
            if name in box_rel:
                related = {k: [" ".join(str(q).lower().split()) for q in box_rel[name].get(k) or []] for k in ("rising", "top")}
            elif goanyapi.key():
                try:
                    related, cost_r = goanyapi.related_queries(name, c["related_geo"])
                except goanyapi.GoAnyError as exc:
                    related_errors.append(f"{name}: {exc}")
            else:
                inbox.request(cfg, state, "related", [name], today)
            c0 += cost_r
            tails[name], searched = split_related(name, related)
            have = {u["phrase"] for u in cands}
            cands += [u for u in searched if u["phrase"] not in have]
        asked += len(cands)
        viewed = screen(c["model"], name, ev[name], cands)
        cosearch += other_names(name, viewed)
        for u in relevant(viewed):
            row = proposed.setdefault(u["phrase"], {**u, "term": u["phrase"], "kind": "generic", "names": []})
            row["names"].append(name)
            row["brand"] |= u["brand"]
            if u["source"] == "searched":
                row["source"] = "searched"
    bench = c["benchmark"].lower()
    vol, c2 = volumes(sorted(proposed), c["location_code"], box_vol)
    inbox.request(cfg, state, "volumes", [p for p in sorted(proposed) if p not in vol], today)
    generic = sorted((dict(r, **vol[p]) for p, r in proposed.items() if vol.get(p, {}).get("volume", 0) >= c["min_volume"]),
                     key=lambda r: -r["volume"])[:c["trends_max"]]
    rows = [{"term": n, "phrase": n, "kind": "name", "names": [], "brand": True, "evidence": "", **vol.get(n, {})} for n in todo] + generic
    series, c3 = fetch_series([r["phrase"] for r in rows], bench, today)
    daily = measure(rows, series, c["min_daily"], today)
    order = {"prioritize": 0, "watch": 1, "skip": 2}
    rows.sort(key=lambda r: (order[r["decision"]], -(r.get("per_day") or 0), -(r.get("volume") or 0)))
    for name in todo:
        s = seen.setdefault(name, {"first": today.isoformat(), "runs": 0})
        if not s.pop("no_ev", False):   # a read without evidence does not count as a read
            s["runs"] += 1
        if not ev[name]["posts"] and not ev[name]["readme"]:
            s["no_ev"] = True
    cutoff = (today - timedelta(days=STATE_DAYS)).isoformat()
    snap.save(path, {n: s for n, s in seen.items() if s["first"] >= cutoff})
    return {"names": todo, "rows": rows, "proposed": asked, "relevant": len(proposed), "kept": len(generic), "benchmark": bench,
            "bench_daily": round(daily) if daily else None, "volume_month": next((r["month"] for r in generic if r.get("month")), ""),
            "no_evidence": [n for n in todo if not ev[n]["posts"] and not ev[n]["readme"]],
            "related": use_related, "tails": {n: v for n, v in tails.items() if v}, "cosearch": cosearch,
            "related_errors": related_errors + trends.last_errors, "llm": llm.provider() if llm.available() else "",
            "trends": trends.provider(), "cost_micros": cost + c0 + c2 + c3}
