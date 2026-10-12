"""The AIsa-free paths: xAI chat, pytrends Trends, the inbox, clean skips. No network."""
import io
import json
from datetime import date

import pytest

from radar import __main__ as cli
from radar import inbox, judge, llm, new_sites, sources, trends, usecases

TODAY = date(2026, 10, 12)
SEARCH = {"data": [{"id": "1", "text": "SynthID Detector tells if an image is AI made", "author_id": "u1",
                    "created_at": "2026-10-11T08:00:00.000Z", "public_metrics": {"like_count": 900}},
                   {"id": "2", "text": "meh", "author_id": "u2", "public_metrics": {"like_count": 3}}],
          "includes": {"users": [{"id": "u1", "username": "alice"}, {"id": "u2", "username": "bob"}]}}


def _write(state, kind, obj, day=None):
    d = state / "inbox" / (day or date.today()).isoformat()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}.json").write_text(json.dumps(obj))


def test_llm_auto_prefers_xai_and_sends_the_grok_model(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "k")
    llm.configure({"model": "grok-x"})
    sent = {}

    class Res(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        sent["auth"] = req.headers["Authorization"]
        return Res(json.dumps({"choices": [{"message": {"content": "[{\"i\": 0, \"score\": 0.9}]"}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert llm.provider() == "xai"
    rows = [{"term": "opus 5.5"}]
    assert judge.score(rows, "gpt-4.1-mini") == {"judged": 1} and rows[0]["named"] == 0.9
    assert sent["url"] == "https://api.x.ai/v1/chat/completions" and sent["body"]["model"] == "grok-x"
    assert sent["auth"] == "Bearer k"


def test_without_any_model_key_the_judge_skips(monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    assert judge.score([{"term": "x"}], "m") == {"judged": 0, "skipped": True}
    assert usecases.propose("m", "n", {"posts": [{"text": "a", "likes": 1, "by": "a"}], "readme": "", "repo": ""}) == []


def test_inbox_turns_x_v2_and_keyword_planner_shapes_into_rows():
    tw = inbox.tweets(SEARCH)
    assert tw[0]["by"] == "alice" and tw[0]["likes"] == 900 and tw[0]["url"] == "https://x.com/alice/status/1"
    assert tw[0]["created"].year == 2026 and tw[1]["created"] is None
    vol = inbox.volumes({"metrics": [{"keyword": "AI Image Detector", "avgMonthlySearches": 74000, "monthlySearchVolumes": [
        {"year": 2026, "month": "JULY", "monthlySearches": 1}, {"year": 2026, "month": "AUGUST", "monthlySearches": 2}]}]})
    assert vol == {"ai image detector": {"volume": 74000, "month": "2026-08"}}


def test_x_new_reads_the_inbox_and_asks_for_the_rest(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    cfg["x_accounts"] = {"labs": ["OpenAI", "xai"]}
    _write(tmp_path, "x_posts", {"users": {"openai": SEARCH}})
    out = sources.x_new(cfg, tmp_path)
    assert out["posts"] == 2 and out["first_run"]
    req = json.loads((inbox.folder(cfg, tmp_path) / "requests.json").read_text())
    assert req["x_users"] == ["xai"]


def test_x_new_skips_without_inbox_or_key(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    cfg["x_accounts"] = {"labs": ["OpenAI"]}
    assert sources.x_new(cfg, tmp_path)["skipped"]


def test_new_sites_skip_without_aisa(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    assert new_sites.new_sites(cfg, tmp_path)["skipped"]


def test_evidence_and_volumes_come_from_the_inbox_without_aisa(monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    ev, cost = usecases.evidence(["synthid detector"], {}, TODAY, {"synthid detector": SEARCH})
    assert cost == 0 and ev["synthid detector"]["posts"][0]["by"] == "alice"
    vol, _ = usecases.volumes(["a", "b"], 2840, {"a": {"volume": 5, "month": "2026-08"}})
    assert vol == {"a": {"volume": 5, "month": "2026-08"}}


class FakeTrends:
    calls = 0

    def __init__(self, fail=0):
        self.fail = fail

    def build_payload(self, kws, timeframe, geo):
        self.kws = kws
        FakeTrends.calls += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("Google returned a response with code 429")

    def interest_over_time(self):
        import pandas as pd
        idx = pd.date_range("2026-10-01", periods=3, freq="D")
        return pd.DataFrame({k: [1 + i, 2 + i, 3 + i] for i, k in enumerate(self.kws)}, index=idx)


def test_pytrends_provider_returns_rows_caches_and_retries(tmp_path, monkeypatch):
    pytest.importorskip("pytrends")
    pytest.importorskip("pandas")
    monkeypatch.delenv("AISA_API_KEY")
    trends.configure({"delay": 0, "backoff": 0, "retries": 2}, tmp_path)
    fake = FakeTrends(fail=1)
    monkeypatch.setattr("pytrends.request.TrendReq", lambda **kw: fake)
    FakeTrends.calls = 0
    out, cost = trends.fetch({"p": (["a", "gpts"], "2026-10-01", "2026-10-03")})
    assert trends.provider() == "pytrends" and cost == 0
    assert out["p"][0] == (date(2026, 10, 1), 1, 2) and FakeTrends.calls == 2
    trends.fetch({"p": (["a", "gpts"], "2026-10-01", "2026-10-03")})
    assert FakeTrends.calls == 2   # served from the cache


def test_pytrends_rate_limit_stops_and_is_reported(tmp_path, monkeypatch):
    pytest.importorskip("pytrends")
    monkeypatch.delenv("AISA_API_KEY")
    trends.configure({"delay": 0, "backoff": 0, "retries": 1, "cache": False}, tmp_path)
    monkeypatch.setattr("pytrends.request.TrendReq", lambda **kw: FakeTrends(fail=99))
    out, _ = trends.fetch({"a": (["x"], "2026-10-01", "2026-10-03"), "b": (["y"], "2026-10-01", "2026-10-03")})
    assert out == {} and any("rate-limited" in e for e in trends.last_errors)


def test_daily_creates_the_output_directory(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({}))
    for mod, fn in ((cli.sources, "x_new"), (cli.sources, "hf_new"), (cli.breakout, "breakouts"), (cli.sitemaps, "diff")):
        monkeypatch.setattr(mod, fn, lambda cfg, state: {"skipped": True})
    out = tmp_path / "deep" / "dir" / "report.md"
    cli.main(["daily", "--config", str(tmp_path / "config.json"), "--state", str(tmp_path / "s"), "--out", str(out),
              "--skip", "newsites,gsc,review,usecases,judge"])
    assert out.exists() and out.with_suffix(".json").exists()
