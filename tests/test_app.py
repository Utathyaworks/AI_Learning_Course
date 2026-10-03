"""Clicks through the real Streamlit app with a fake model and local storage."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streamlit.testing.v1 import AppTest  # noqa: E402

from coach.coding import load_task  # noqa: E402

APP = os.path.join(ROOT, "app.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_MODE_NO_AUTH", "1")
    monkeypatch.setenv("COACH_FAKE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("LOCAL_STATE_PATH", str(tmp_path / "state.json"))
    return tmp_path


def _nav(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, at.exception


def test_gate_blocks_without_login(tmp_path, monkeypatch):
    monkeypatch.delenv("DEV_MODE_NO_AUTH", raising=False)
    monkeypatch.setenv("LOCAL_STATE_PATH", str(tmp_path / "s.json"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("Sign-in isn't set up" in e.value for e in at.error)
    assert not at.sidebar.radio


def test_full_week_flow(env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Dashboard"
    assert "W00" in at.subheader[0].value
    # progress log: hours + note for today, stored for the monitor
    at.number_input(key="log_h").set_value(5.5).run()
    at.text_input(key="log_n").set_value("Linux + Git basics").run()
    next(b for b in at.button if b.label == "Log today").click().run()
    assert not at.exception, at.exception

    for week in ("W00", "W01"):
        _nav(at, "Weekly exam")
        at.selectbox[0].set_value(week).run()
        next(b for b in at.button if b.label == "Start exam").click().run()
        assert not at.exception, at.exception
        radios = [r for r in at.radio if r.key and (r.key.startswith("a_") or r.key.startswith("s_"))]
        assert len([r for r in radios if r.key.startswith("a_")]) == 12
        for r in radios:
            r.set_value(next(i for i, o in enumerate(r.options) if str(o).startswith("Correct")))
        for ta in at.text_area:
            tid = ta.key.split("_", 2)[2]
            ta.set_value(load_task(tid).solution)
        next(b for b in at.button if b.label == "Submit exam").click().run()
        assert not at.exception, at.exception
        assert any(s.value.startswith("Result: 100%") for s in at.subheader), [s.value for s in at.subheader]

    _nav(at, "Review & plan")
    assert any("Advance as planned" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Accept").click().run()
    assert not at.exception, at.exception

    _nav(at, "Dashboard")
    assert any(s.value.startswith("Next exam: W01B") for s in at.subheader), [s.value for s in at.subheader]
    import json
    saved = json.load(open(env / "state.json", encoding="utf-8"))
    assert saved["daily_log"] and saved["daily_log"][0]["hours"] == 5.5
    _nav(at, "Review queue")
    _nav(at, "Settings & cost")
    assert os.path.exists(env / "state.json")
