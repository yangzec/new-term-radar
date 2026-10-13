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


class Done:
    def __init__(self, out, code=0, err=""):
        self.stdout, self.returncode, self.stderr = out, code, err


def _cli(monkeypatch, out, code=0):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        return Done(out, code)

    monkeypatch.setattr(llm, "_cli_path", lambda: "/bin/grok-cli")
    monkeypatch.setattr(llm, "cli_ready", lambda: True)
    monkeypatch.setattr(llm.subprocess, "run", run)
    return seen


def test_auto_order_xai_then_grok_cli_then_aisa_then_nothing(monkeypatch):
    monkeypatch.setattr(llm, "cli_ready", lambda: True)
    monkeypatch.setenv("XAI_API_KEY", "k")
    assert llm.provider() == "xai"
    monkeypatch.delenv("XAI_API_KEY")
    assert llm.provider() == "grok-cli"
    monkeypatch.setattr(llm, "cli_ready", lambda: False)
    assert llm.provider() == "aisa"
    monkeypatch.delenv("AISA_API_KEY")
    assert llm.provider() == "" and not llm.available()


def test_grok_cli_judge_passes_system_prompt_and_model_and_parses_fenced_json(monkeypatch):
    llm.configure({"provider": "grok-cli", "cli_model": "grok-4.6", "cli_timeout": 60})
    answer = 'Sure:\n```json\n[{"i": 0, "score": 0.95}, {"i": 1, "score": 0.1}]\n```'
    seen = _cli(monkeypatch, json.dumps({"ok": True, "data": {"output_text": answer}}))
    rows = [{"term": "opus 5.5"}, {"term": "dashboard"}]
    assert judge.score(rows, "ignored") == {"judged": 2}
    assert rows[0]["named"] == 0.95 and rows[1]["named"] == 0.1
    cmd = seen["cmd"]
    assert cmd[:2] == ["/bin/grok-cli", "chat"] and "--json" in cmd and "--no-web-search" in cmd
    assert cmd[cmd.index("--model") + 1] == "grok-4.6" and cmd[cmd.index("--system") + 1].startswith("You sort terms")
    assert '"opus 5.5"' in cmd[cmd.index("--prompt") + 1] and seen["timeout"] == 90


def test_grok_cli_failure_is_an_error_not_a_crash(monkeypatch):
    llm.configure({"provider": "grok-cli"})
    _cli(monkeypatch, json.dumps({"ok": False, "error": "relogin required"}), code=1)
    rows = [{"term": "x"}]
    out = judge.score(rows, "m")
    assert "relogin" in out["error"] and "named" not in rows[0]
    _cli(monkeypatch, "not json at all", code=2)
    assert usecases.screen("m", "n", {"posts": [], "readme": "r"}, [{"phrase": "a b", "brand": False}]) == [{"phrase": "a b", "brand": False}]


def test_grok_cli_timeout(monkeypatch):
    llm.configure({"provider": "grok-cli", "cli_timeout": 1})
    monkeypatch.setattr(llm, "_cli_path", lambda: "/bin/grok-cli")

    def run(cmd, **kw):
        raise llm.subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(llm.subprocess, "run", run)
    with pytest.raises(llm.LLMError, match="timed out"):
        llm.chat("m", "s", "u")


def test_cli_ready_reads_status_json(monkeypatch):
    llm._cli_ok.clear()
    monkeypatch.undo()   # drop conftest's cli_ready stub for this test
    monkeypatch.setattr(llm, "_cli_path", lambda: "/bin/fake-grok")
    monkeypatch.setattr(llm.subprocess, "run", lambda cmd, **kw: Done(json.dumps({"ok": True, "data": {"logged_in": True}})))
    assert llm.cli_ready()
    llm._cli_ok.clear()
    monkeypatch.setattr(llm.subprocess, "run", lambda cmd, **kw: Done(json.dumps({"ok": True, "data": {"logged_in": False}})))
    assert not llm.cli_ready()
    llm._cli_ok.clear()


def _uc_fakes(monkeypatch):
    monkeypatch.setattr(usecases, "propose", lambda model, name, ev: [{"phrase": "ai phone agent", "evidence": "e", "brand": False}]
                        if ev["posts"] else [])
    monkeypatch.setattr(usecases, "screen", lambda model, name, ev, cands: cands)
    monkeypatch.setattr(usecases, "fetch_series", lambda keywords, bench, today: ({}, 0))
    monkeypatch.setattr(usecases, "_readme", lambda repo: "")


def test_inbox_evidence_for_an_earlier_name_is_used_and_the_read_is_redone(cfg, tmp_path, monkeypatch):
    """Run 1 reads a name with no evidence and asks for it; run 2 (other leads that day)
    still reads it from the inbox, and the volume of its phrase comes from volumes.json."""
    monkeypatch.delenv("AISA_API_KEY")
    _uc_fakes(monkeypatch)
    first = usecases.run(cfg, tmp_path, ["esp32 iphone agent"], {}, TODAY)
    assert first["no_evidence"] == ["esp32 iphone agent"]
    assert json.loads((inbox.folder(cfg, tmp_path, TODAY) / "requests.json").read_text())["x_search"] == ["esp32 iphone agent"]
    _write(tmp_path, "x_posts", {"search": {"ESP32 iPhone Agent": SEARCH}}, TODAY)
    _write(tmp_path, "volumes", {"metrics": [{"keyword": "ai phone agent", "avgMonthlySearches": 1600,
                                              "monthlySearchVolumes": [{"year": 2026, "month": "AUGUST", "monthlySearches": 1900}]}]}, TODAY)
    second = usecases.run(cfg, tmp_path, ["some other lead"], {}, TODAY)
    assert second["names"][0] == "esp32 iphone agent" and second["no_evidence"] == ["some other lead"]
    row = next(r for r in second["rows"] if r["phrase"] == "ai phone agent")
    assert row["volume"] == 1600 and row["month"] == "2026-08"
    state = usecases.snap.load(tmp_path / "usecases.json.gz")
    assert state["esp32 iphone agent"] == {"first": TODAY.isoformat(), "runs": 1}


def test_requests_never_ask_for_what_the_inbox_holds(cfg, tmp_path):
    _write(tmp_path, "x_posts", {"users": {"openai": SEARCH}}, TODAY)
    _write(tmp_path, "volumes", {"metrics": [{"keyword": "ai phone agent", "avgMonthlySearches": 1}]}, TODAY)
    inbox.request(cfg, tmp_path, "x_users", ["OpenAI", "xai"], TODAY)
    inbox.request(cfg, tmp_path, "volumes", ["ai phone agent", "run llm on mac"], TODAY)
    req = json.loads((inbox.folder(cfg, tmp_path, TODAY) / "requests.json").read_text())
    assert req == {"x_users": ["xai"], "volumes": ["run llm on mac"]}


def test_requests_are_pruned_even_when_nothing_new_is_asked(cfg, tmp_path):
    inbox.request(cfg, tmp_path, "x_search", ["tubetext"], TODAY)
    _write(tmp_path, "x_posts", {"search": {"tubetext": SEARCH}}, TODAY)
    inbox.request(cfg, tmp_path, "x_search", [], TODAY)
    assert json.loads((inbox.folder(cfg, tmp_path, TODAY) / "requests.json").read_text()) == {"x_search": []}


def test_incomplete_banner_flags_our_providers_gaps():
    base = {"names": ["n"], "llm": "grok-cli"}
    assert not usecases.degraded(base)
    assert usecases.degraded(dict(base, trends_errors=["pytrends: 429"]))
    assert usecases.degraded(dict(base, llm=""))
    assert usecases.degraded(dict(base, inbox_missing={"volumes": ["a b"]}))


def test_keywords_on_request_reads_the_inbox_and_leaves_state_alone(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("AISA_API_KEY")
    monkeypatch.setattr(usecases.aisa, "has_key", lambda: False)
    _uc_fakes(monkeypatch)
    _write(tmp_path, "x_posts", {"search": {"d2r mobile": SEARCH}}, TODAY)
    uc = usecases.run(cfg, tmp_path, ["d2r mobile"], {}, TODAY, on_request=True)
    assert uc["names"] == ["d2r mobile"] and uc["no_evidence"] == [] and uc["inbox_missing"] == {"volumes": ["ai phone agent"]}
    assert not (tmp_path / "usecases.json.gz").exists()


def test_has_key_never_raises(monkeypatch, tmp_path):
    from radar import aisa
    monkeypatch.delenv("AISA_API_KEY")
    monkeypatch.setattr(aisa.Path, "home", lambda: tmp_path)
    assert aisa.has_key() is False
    monkeypatch.setenv("AISA_API_KEY", "k")
    assert aisa.has_key() is True


class EmptyTrends(FakeTrends):
    def __init__(self, empties):
        super().__init__()
        self.empties = empties

    def interest_over_time(self):
        import pandas as pd
        if self.empties:
            self.empties -= 1
            return pd.DataFrame()
        return super().interest_over_time()


def test_pytrends_empty_series_is_retried_once_then_a_failure_and_never_cached(tmp_path, monkeypatch):
    pytest.importorskip("pytrends")
    monkeypatch.delenv("AISA_API_KEY")
    monkeypatch.setattr(trends.aisa, "has_key", lambda: False)
    trends.configure({"delay": 0, "backoff": 0, "retries": 2}, tmp_path)
    FakeTrends.calls = 0
    monkeypatch.setattr("pytrends.request.TrendReq", lambda **kw: EmptyTrends(empties=5))
    out, _ = trends.fetch({"s": (["d2r mobile"], "2026-10-01", "2026-10-03")})
    assert out == {} and FakeTrends.calls == 2 and "empty series" in trends.last_errors[0]
    assert not list((tmp_path / "trends_cache").rglob("*.json"))
    monkeypatch.setattr("pytrends.request.TrendReq", lambda **kw: EmptyTrends(empties=1))
    out, _ = trends.fetch({"s": (["d2r mobile"], "2026-10-01", "2026-10-03")})
    assert out["s"][0] == (date(2026, 10, 1), 1) and not trends.last_errors


def test_an_empty_series_in_the_cache_is_not_trusted(tmp_path, monkeypatch):
    pytest.importorskip("pytrends")
    monkeypatch.delenv("AISA_API_KEY")
    monkeypatch.setattr(trends.aisa, "has_key", lambda: False)
    trends.configure({"delay": 0, "backoff": 0}, tmp_path)
    cp = trends._cache_path(["x"], "2026-10-01", "2026-10-03")
    cp.parent.mkdir(parents=True)
    cp.write_text("[]")
    monkeypatch.setattr("pytrends.request.TrendReq", lambda **kw: FakeTrends())
    out, _ = trends.fetch({"s": (["x"], "2026-10-01", "2026-10-03")})
    assert out["s"]


def test_pytrends_window_ends_before_the_unsettled_days():
    assert trends.settled_window("2026-10-14", date(2026, 10, 14)) == "2026-10-11"
    assert trends.settled_window("2026-09-01", date(2026, 10, 14)) == "2026-09-01"
