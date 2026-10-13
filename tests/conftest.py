import copy

import pytest

from radar import config, i18n


@pytest.fixture(autouse=True)
def english(monkeypatch):
    i18n.set_lang("en")
    monkeypatch.delenv("GOANYAPI_API_KEY", raising=False)   # the optional source is off unless a test turns it on
    monkeypatch.setenv("AISA_API_KEY", "test")              # network calls are faked; providers resolve to AIsa
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    from radar import llm, trends
    llm.configure({})
    monkeypatch.setattr(llm, "cli_ready", lambda: False)   # a real grok-cli on the box must not be picked
    trends.configure({})


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(config.DEFAULTS)
    c["_dir"] = tmp_path
    return c
