"""The study plan: a list of weekly entries whose dates follow from their order.

Dates are never stored per entry. Each entry lasts `days` (default 7) and the
next one starts the day after, so inserting a week shifts everything after it.
"""
from __future__ import annotations

import copy
import datetime as dt
from typing import Any

import yaml

KINDS = {"tooling", "topic", "depth", "consolidation", "capstone", "extra", "rest"}
NO_EXAM = {"rest"}


def load_base_plan(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    # YAML turns ISO dates into date objects; keep strings so the state stays JSON.
    for k in ("start_date", "max_end_date"):
        if k in plan:
            plan[k] = str(plan[k])
    for e in plan["entries"]:
        e.setdefault("days", 7)
        e.setdefault("math", [])
        e.setdefault("build", [])
        e.setdefault("key_results", list(e["math"]))
        e.setdefault("tasks", [])
        e.setdefault("notes", [])
        e.setdefault("focus", [])
        e.setdefault("status", "todo")
        e.setdefault("phase", 0)
        assert e["kind"] in KINDS, e
    return plan


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def schedule(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Entries with computed `start` and `end` dates (inclusive)."""
    out, cur = [], _d(plan["start_date"])
    for e in plan["entries"]:
        start = cur
        end = start + dt.timedelta(days=int(e.get("days", 7)) - 1)
        out.append({**e, "start": start, "end": end})
        cur = end + dt.timedelta(days=1)
    return out


def end_date(plan) -> dt.date:
    return schedule(plan)[-1]["end"]


def max_end(plan) -> dt.date:
    return _d(plan.get("max_end_date", "2028-01-16"))


def find(plan, entry_id) -> dict[str, Any]:
    for e in plan["entries"]:
        if e["id"] == entry_id:
            return e
    raise KeyError(entry_id)


def index_of(plan, entry_id) -> int:
    for i, e in enumerate(plan["entries"]):
        if e["id"] == entry_id:
            return i
    raise KeyError(entry_id)


def next_entry(plan, entry_id):
    i = index_of(plan, entry_id)
    return plan["entries"][i + 1] if i + 1 < len(plan["entries"]) else None


def next_consolidation(plan, entry_id):
    i = index_of(plan, entry_id)
    for e in plan["entries"][i + 1 :]:
        if e["kind"] in ("consolidation", "extra"):
            return e
    return None


def current_entry(plan) -> dict[str, Any]:
    """The first entry whose exam has not been taken or skipped (rest weeks have no exam)."""
    for e in schedule(plan):
        if e["status"] not in ("done", "skipped") and e["kind"] not in NO_EXAM:
            return e
    return schedule(plan)[-1]


def exam_ids(plan) -> list[str]:
    return [e["id"] for e in plan["entries"] if e["kind"] not in NO_EXAM]


def next_depth(plan, entry_id):
    """The depth (or consolidation) week that follows an entry: where carried-over work goes."""
    i = index_of(plan, entry_id)
    for e in plan["entries"][i + 1 :]:
        if e["kind"] in ("depth", "consolidation", "extra"):
            return e
    return None


def entry_on(plan, day: dt.date):
    for e in schedule(plan):
        if e["start"] <= day <= e["end"]:
            return e
    return None


def _log(state, change: str, detail: str, today: dt.date | None = None):
    state["plan_history"].append(
        {"date": (today or dt.date.today()).isoformat(), "change": change, "detail": detail}
    )


# ---- edits -------------------------------------------------------------------

def add_note(state, entry_id: str, text: str, today=None):
    find(state["plan"], entry_id)["notes"].append(text)
    _log(state, "note", f"{entry_id}: {text}", today)


def add_focus(state, entry_id: str, topics: list[str], today=None):
    e = find(state["plan"], entry_id)
    for t in topics:
        if t not in e["focus"]:
            e["focus"].append(t)
    _log(state, "focus", f"{entry_id}: relearn {', '.join(topics)}", today)


def insert_extra_week(state, after_id: str, topics: list[str], today=None) -> str:
    plan = state["plan"]
    n = 1 + sum(1 for e in plan["entries"] if e["kind"] == "extra")
    new = {
        "id": f"X{n}",
        "kind": "extra",
        "phase": find(plan, after_id).get("phase", 0),
        "title": f"Extra consolidation X{n} — relearn and re-test",
        "days": 7,
        "math": ["Relearning protocol on: " + (", ".join(topics) or "weakest topics")],
        "build": ["Re-implement the weak topics' modules from blank files", "External problem set on the same topics"],
        "deliverable": "Every listed topic re-tested at 80% or more.",
        "key_results": [],
        "tasks": [],
        "notes": [],
        "focus": list(topics),
        "status": "todo",
    }
    plan["entries"].insert(index_of(plan, after_id) + 1, new)
    _log(state, "insert", f"{new['id']} after {after_id} ({', '.join(topics)})", today)
    return new["id"]


def set_days(state, entry_id: str, days: int, today=None, why: str = ""):
    e = find(state["plan"], entry_id)
    old = int(e.get("days", 7))
    days = max(1, min(28, int(days)))
    e["days"] = days
    _log(state, "reschedule", f"{entry_id}: {old} → {days} days" + (f" ({why})" if why else ""), today)


def insert_rest(state, after_id: str, days: int, today=None) -> str:
    plan = state["plan"]
    n = 1 + sum(1 for e in plan["entries"] if e["kind"] == "rest")
    while any(e["id"] == f"R{n}" for e in plan["entries"]):
        n += 1
    new = {"id": f"R{n}", "kind": "rest", "phase": find(plan, after_id).get("phase", 0),
           "title": f"Rest — {days} day{'s' if days != 1 else ''} off", "days": int(days),
           "math": [], "build": [], "deliverable": "Rested. Optional: at most 30 minutes a day on due review items.",
           "key_results": [], "tasks": [], "notes": [], "focus": [], "status": "todo"}
    plan["entries"].insert(index_of(plan, after_id) + 1, new)
    _log(state, "insert", f"{new['id']} ({days} rest days) after {after_id}", today)
    return new["id"]


def snapshot(state, label: str, today=None):
    state.setdefault("snapshots", []).append(
        {"label": label, "date": (today or dt.date.today()).isoformat(), "plan": copy.deepcopy(state["plan"])})
    state["snapshots"] = state["snapshots"][-10:]


def remove_entry(state, entry_id: str, today=None):
    plan = state["plan"]
    e = find(plan, entry_id)
    if e["kind"] not in ("extra", "rest"):
        raise ValueError("Only extra consolidation weeks and added rest days can be removed; edit the others instead.")
    plan["entries"].remove(e)
    _log(state, "remove", entry_id, today)


def move_entry(state, entry_id: str, delta: int, today=None):
    plan = state["plan"]
    i = index_of(plan, entry_id)
    j = max(0, min(len(plan["entries"]) - 1, i + delta))
    if i == j:
        return
    if plan["entries"][j]["status"] == "done" or plan["entries"][i]["status"] == "done":
        raise ValueError("Finished weeks can't be moved.")
    e = plan["entries"].pop(i)
    plan["entries"].insert(j, e)
    _log(state, "move", f"{entry_id} {'later' if delta > 0 else 'earlier'} by {abs(delta)}", today)


def skippable_targets(plan) -> list[str]:
    """Week ids strictly after the current one — valid targets for a placement jump."""
    i = index_of(plan, current_entry(plan)["id"])
    return [e["id"] for e in plan["entries"][i + 1 :]]


def entries_between(plan, target_id: str) -> list[dict[str, Any]]:
    """Exam-bearing entries from the current one up to (not including) target_id."""
    i_cur = index_of(plan, current_entry(plan)["id"])
    i_tgt = index_of(plan, target_id)
    if i_tgt <= i_cur:
        raise ValueError("Pick a week after where you are now.")
    return [e for e in plan["entries"][i_cur:i_tgt] if e["kind"] not in NO_EXAM]


def skip_to(state, target_id: str, skipped_ids: list[str], today=None):
    """After a passed placement check: mark the skipped weeks and let target_id become current."""
    plan = state["plan"]
    for eid in skipped_ids:
        find(plan, eid)["status"] = "skipped"
    _log(state, "skip", f"Placement check passed — skipped {', '.join(skipped_ids)} to start at {target_id}", today)


def edit_entry(state, entry_id: str, fields: dict[str, Any], today=None):
    e = find(state["plan"], entry_id)
    changed = []
    for k, v in fields.items():
        if k in {"title", "math", "build", "deliverable", "days", "notes", "focus"} and e.get(k) != v:
            e[k] = copy.deepcopy(v)
            changed.append(k)
    if changed:
        _log(state, "edit", f"{entry_id}: {', '.join(changed)}", today)


def ensure_plan(state, base_plan_path: str):
    if not state.get("plan"):
        state["plan"] = load_base_plan(base_plan_path)
        _log(state, "create", "Plan created from data/plan.yaml")
    return state["plan"]
