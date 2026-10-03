import datetime as dt
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coach import monitor, review, storage  # noqa: E402
from coach import exam as examlib  # noqa: E402
from coach import plan as planlib  # noqa: E402
from coach.config import Settings  # noqa: E402
from coach.fake_llm import FakeClient  # noqa: E402

BASE = os.path.join(ROOT, "data", "plan.yaml")


def fresh():
    s = storage.new_state()
    planlib.ensure_plan(s, BASE)
    return s


def sched(s):
    return {e["id"]: e for e in planlib.schedule(s["plan"])}


def test_rest_weeks_have_no_exam_and_are_skipped():
    s = fresh()
    assert "R1" not in planlib.exam_ids(s["plan"])
    for e in s["plan"]["entries"]:
        if e["id"] == "R1":
            break
        e["status"] = "done"
    assert planlib.current_entry(s["plan"])["id"] == "D4"
    with pytest.raises(ValueError):
        examlib.build_exam(Settings(openai_api_key="x"), s, "R1", dt.date(2026, 12, 30), client=FakeClient())


def test_depth_exam_uses_covered_weeks_tasks():
    s = fresh()
    s["attempts"] = [{"entry": "W01", "by_topic": {"QR": 0.3}, "coding": {"items": [
        {"task_id": "w01_gram_schmidt", "score": 0.2}, {"task_id": "w01_broadcast_shape", "score": 1.0},
        {"task_id": "w01b_lines_polygons", "score": 1.0}, {"task_id": "w01b_rotations_camera", "score": 0.5}]}}]
    ex = examlib.build_exam(Settings(openai_api_key="x"), s, "D1", dt.date(2026, 10, 25), client=FakeClient())
    ids = [t["id"] for t in ex["tasks"]]
    covered = set(planlib.find(s["plan"], "W01")["tasks"]) | set(planlib.find(s["plan"], "W01B")["tasks"])
    assert len(ids) == 2 and set(ids) <= covered
    assert set(ids) == {"w01_gram_schmidt", "w01b_rotations_camera"}  # the two weakest of the covered weeks


def test_overdue_week_proposes_reschedule_and_nothing_changes_until_accepted():
    s = fresh()
    end_before = planlib.end_date(s["plan"])
    w00_end = sched(s)["W00"]["end"]
    today = w00_end + dt.timedelta(days=5)
    new = monitor.check(s, today)
    ext = [p for p in new if p["type"] == monitor.EXTEND]
    assert ext and ext[0]["source"] == "W00" and ext[0]["status"] == "pending"
    assert planlib.end_date(s["plan"]) == end_before  # proposal only
    review.apply(s, ext[0], today)
    assert planlib.end_date(s["plan"]) == end_before + dt.timedelta(days=7)
    assert sched(s)["W00"]["end"] == today + dt.timedelta(days=2)
    # no duplicate proposal the same week
    assert not [p for p in monitor.check(s, today) if p["type"] == monitor.EXTEND]


def test_undo_restores_the_plan():
    s = fresh()
    before = planlib.end_date(s["plan"])
    today = sched(s)["W00"]["end"] + dt.timedelta(days=4)
    p = [p for p in monitor.check(s, today) if p["type"] == monitor.EXTEND][0]
    review.apply(s, p, today)
    assert planlib.end_date(s["plan"]) != before
    review.undo(s, p, today)
    assert planlib.end_date(s["plan"]) == before and p["status"] == "undone"
    assert s["plan_history"][-1]["change"] == "undo"


def test_behind_pace_offers_extend_or_move_to_depth_week():
    s = fresh()
    for e in s["plan"]["entries"][:2]:
        e["status"] = "done"  # W00, W01 done → current is W01B
    w = sched(s)["W01B"]
    today = w["end"] - dt.timedelta(days=1)
    s["daily_log"] = [{"date": (w["start"] + dt.timedelta(days=i)).isoformat(), "entry": "W01B", "hours": 2, "note": ""} for i in range(5)]
    new = monitor.check(s, today)
    kinds = {p["type"] for p in new}
    assert {monitor.EXTEND, monitor.MOVE_STRETCH} <= kinds
    mv = [p for p in new if p["type"] == monitor.MOVE_STRETCH][0]
    assert mv["payload"]["target"] == "D1"
    end = planlib.end_date(s["plan"])
    review.apply(s, mv, today)
    assert planlib.end_date(s["plan"]) == end  # restructure keeps the dates
    assert any("Carried over from W01B" in n for n in planlib.find(s["plan"], "D1")["notes"])


def test_ahead_proposes_pull_forward():
    s = fresh()
    w = sched(s)["W01"]
    s["plan"]["entries"][0]["status"] = "done"
    planlib.find(s["plan"], "W01")["status"] = "done"
    s["attempts"] = [{"id": "a1", "entry": "W01", "total": 0.9, "by_topic": {}, "coding": {"items": []}}]
    s["checklist"]["W01"] = monitor.items(planlib.find(s["plan"], "W01"))
    today = w["start"] + dt.timedelta(days=3)
    p = [p for p in monitor.check(s, today) if p["type"] == monitor.PULL_FORWARD][0]
    review.apply(s, p, today)
    assert sched(s)["W01B"]["start"] == today + dt.timedelta(days=1)


def test_overload_proposes_rest_and_max_end_is_enforced():
    s = fresh()
    today = dt.date(2026, 10, 20)
    s["daily_log"] = [{"date": (today - dt.timedelta(days=i)).isoformat(), "entry": "D1", "hours": 7, "note": ""} for i in range(14)]
    p = [p for p in monitor.check(s, today) if p["type"] == monitor.ADD_REST][0]
    s["plan"]["max_end_date"] = planlib.end_date(s["plan"]).isoformat()
    with pytest.raises(ValueError):
        review.apply(s, p, today)
    assert p["status"] == "pending" and not any(e["id"] == "R3" for e in s["plan"]["entries"])
    s["plan"]["max_end_date"] = "2028-01-16"
    review.apply(s, p, today)
    assert any(e["kind"] == "rest" and e["days"] == 3 for e in s["plan"]["entries"] if e["id"] not in ("R1", "R2"))
