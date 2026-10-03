"""Progress monitoring: turns the daily log, the weekly checklist and exam scores into
reschedule / restructure proposals. Nothing here changes the plan — every proposal waits
for the learner to accept, edit or reject it on the Review & plan page, and accepted
changes can be undone.
"""
from __future__ import annotations

import datetime as dt
import uuid

from . import plan as planlib

EXTEND, PULL_FORWARD, MOVE_STRETCH, ADD_REST = "extend", "pull_forward", "move_stretch", "add_rest"
MONITOR_TYPES = {EXTEND, PULL_FORWARD, MOVE_STRETCH, ADD_REST}
HOURS_PER_STUDY_DAY = 6.0  # fallback when a state predates per-weekday targets
DEFAULT_DAILY_HOURS = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0]  # Mon..Sun


def daily_target(state, weekday: int) -> float:
    hours = state.get("daily_hours") or DEFAULT_DAILY_HOURS
    return float(hours[weekday]) if 0 <= weekday < len(hours) else 0.0


def planned_hours(state, entry, today: dt.date) -> float:
    """Sum of this week's per-weekday targets from its start up to and including today."""
    end = min(today, entry["end"])
    total, d = 0.0, entry["start"]
    while d <= end:
        total += daily_target(state, d.weekday())
        d += dt.timedelta(days=1)
    return total


def items(entry) -> list[str]:
    """The checklist for a week: its math and build items."""
    return [f"Math: {m}" for m in entry.get("math", [])] + [f"Build: {b}" for b in entry.get("build", [])]


def completion(state, entry) -> float:
    its = items(entry)
    if not its:
        return 1.0
    done = set(state.get("checklist", {}).get(entry["id"], []))
    return sum(i in done for i in its) / len(its)


def study_days_elapsed(entry, today: dt.date) -> int:
    """Mon–Sat days of the entry up to and including today (Sunday is review + blog)."""
    end = min(today, entry["end"])
    n, d = 0, entry["start"]
    while d <= end:
        n += d.weekday() != 6
        d += dt.timedelta(days=1)
    return n


def hours_logged(state, start: dt.date, end: dt.date) -> float:
    return sum(float(r.get("hours", 0)) for r in state.get("daily_log", [])
               if start.isoformat() <= r["date"] <= end.isoformat())


def study_streak(state, today: dt.date) -> int:
    """Consecutive days, ending today or yesterday, with a logged study session."""
    logged = {r["date"] for r in state.get("daily_log", []) if float(r.get("hours", 0)) > 0}
    streak, d = 0, today if today.isoformat() in logged else today - dt.timedelta(days=1)
    while d.isoformat() in logged:
        streak += 1
        d -= dt.timedelta(days=1)
    return streak


def pace_signal(state, entry, today: dt.date) -> dict | None:
    """Where this week's human-logged checklist and hours stand right now — a live readout,
    not just the once-a-day automatic check. None when the entry has no checklist to track."""
    its = items(entry)
    if not its:
        return None
    done = sum(1 for i in its if i in set(state.get("checklist", {}).get(entry["id"], [])))
    comp = completion(state, entry)
    planned = planned_hours(state, entry, today)
    logged = hours_logged(state, entry["start"], min(today, entry["end"]))
    has_log = any(entry["start"].isoformat() <= r["date"] <= today.isoformat() for r in state.get("daily_log", []))
    low_hours = has_log and planned > 0 and logged < 0.5 * planned
    why = f"{done} of {len(its)} checklist items done ({100 * comp:.0f}%)" + (
        f", {logged:.0f} of ~{planned:.0f} planned hours logged" if has_log else "")
    return {"done": done, "total": len(its), "completion": comp, "planned": planned, "logged": logged,
            "has_log": has_log, "low_hours": low_hours, "behind": comp < 0.6 or low_hours, "why": why}


def pace_proposals(state, entry, today: dt.date) -> list[dict]:
    """Build (but don't record) the extend / move-stretch proposals for a behind-pace entry."""
    sig = pace_signal(state, entry, today)
    if not sig or not sig["behind"]:
        return []
    out = [_prop(entry["id"], EXTEND, f"Reschedule: add 3 days to {entry['id']}",
                 f"{sig['why']}. Extend {entry['id']} by 3 days; later weeks move by 3 days.",
                 {"entry": entry["id"], "days": int(entry.get("days", 7)) + 3}, today)]
    tgt = planlib.next_depth(state["plan"], entry["id"])
    if tgt and entry["kind"] in ("topic", "tooling"):
        left = [i for i in items(entry) if i not in set(state.get("checklist", {}).get(entry["id"], []))]
        out.append(_prop(entry["id"], MOVE_STRETCH, f"Restructure: move unfinished items from {entry['id']} to {tgt['id']}",
                         f"{sig['why']}. Keep the dates; take the exam on time, and do these in depth week {tgt['id']}: "
                         + "; ".join(left[:6]) + ("…" if len(left) > 6 else ""),
                         {"entry": entry["id"], "target": tgt["id"], "items": left}, today))
    return out


def propose_now(state, entry, today: dt.date) -> list[dict]:
    """On-demand pace check — same proposals as the automatic one, callable any day the learner
    asks for it, not only in the last 2 days of the week. Still just proposes: nothing changes
    until it's accepted on Review & plan."""
    new = [p for p in pace_proposals(state, entry, today) if not _recent(state, p["type"], p["source"], today)]
    state["proposals"].extend(new)
    return new


def _prop(entry_id, kind, title, detail, payload, today):
    return {"id": uuid.uuid4().hex[:8], "created": today.isoformat(), "source": entry_id, "type": kind,
            "title": title, "detail": detail, "payload": payload, "status": "pending", "by": "monitor"}


def _recent(state, kind, entry_id, today, days=7) -> bool:
    cutoff = (today - dt.timedelta(days=days)).isoformat()
    return any(p["type"] == kind and p["source"] == entry_id and p["created"] >= cutoff for p in state["proposals"])


def check(state, today: dt.date) -> list[dict]:
    """Return new proposals (also appended to state['proposals']). Cheap: no API calls."""
    plan = state["plan"]
    sched = {e["id"]: e for e in planlib.schedule(plan)}
    cur = sched[planlib.current_entry(plan)["id"]]
    new = []

    def add(p):
        if not _recent(state, p["type"], p["source"], today):
            new.append(p)

    # 1. Overdue: the week ended more than 2 days ago and its exam is not taken.
    if cur["status"] != "done" and today > cur["end"] + dt.timedelta(days=2):
        extra = (today - cur["end"]).days + 2
        add(_prop(cur["id"], EXTEND, f"Reschedule: give {cur['id']} {extra} more days",
                  f"{cur['id']} ended on {cur['end']:%a %d %b} but its exam isn't done. Extend it by {extra} days so it "
                  f"ends {today + dt.timedelta(days=2):%a %d %b}; every later week moves by {extra} days "
                  f"(plan end {planlib.end_date(plan) + dt.timedelta(days=extra):%d %b %Y}).",
                  {"entry": cur["id"], "days": int(cur.get("days", 7)) + extra}, today))

    # 2. Behind pace near the end of a week: low checklist completion or low hours.
    elif cur["start"] <= today <= cur["end"] and (cur["end"] - today).days <= 2:
        for p in pace_proposals(state, cur, today):
            add(p)

    # 3. Ahead: the latest finished week scored high and still has 2+ days left.
    done = [a for a in state["attempts"] if a.get("entry") in sched]
    if done:
        last = done[-1]
        e = sched[last["entry"]]
        if last["total"] >= 0.85 and completion(state, e) >= 0.9 and today <= e["end"] - dt.timedelta(days=2):
            new_days = max(3, (today - e["start"]).days + 1)
            add(_prop(e["id"], PULL_FORWARD, f"Reschedule: start {planlib.next_entry(plan, e['id'])['id'] if planlib.next_entry(plan, e['id']) else 'the next week'} early",
                      f"{e['id']} is finished ({100 * last['total']:.0f}% on the exam, checklist done) with "
                      f"{(e['end'] - today).days} days left. Shorten it to {new_days} days so the next week starts tomorrow; "
                      f"the plan ends {(e['end'] - today).days} days earlier.",
                      {"entry": e["id"], "days": new_days}, today))

    # 4. Fatigue: far more hours than planned for two weeks, or three falling exam scores.
    start14 = today - dt.timedelta(days=13)
    h14 = hours_logged(state, start14, today)
    totals = [a["total"] for a in state["attempts"][-3:]]
    falling = len(totals) == 3 and totals[0] > totals[1] > totals[2] and totals[2] < 0.7
    if h14 > 90 or falling:
        why = f"{h14:.0f} hours logged in 14 days (plan: ~72)" if h14 > 90 else \
            "three exam scores in a row have fallen (" + " → ".join(f"{100 * t:.0f}%" for t in totals) + ")"
        add(_prop(cur["id"], ADD_REST, f"Add 3 rest days after {cur['id']}",
                  f"{why}. Insert 3 rest days after {cur['id']}; later weeks move by 3 days.",
                  {"after": cur["id"], "days": 3}, today))

    state["proposals"].extend(new)
    return new


def apply(state, p: dict, today: dt.date):
    pl = p["payload"]
    if p["type"] in (EXTEND, PULL_FORWARD):
        planlib.set_days(state, pl["entry"], pl["days"], today, why=p["title"])
    elif p["type"] == MOVE_STRETCH:
        planlib.add_note(state, pl["target"], f"Carried over from {pl['entry']}: " + "; ".join(pl["items"]), today)
        planlib.add_note(state, pl["entry"], f"Unfinished items moved to {pl['target']}.", today)
    elif p["type"] == ADD_REST:
        planlib.insert_rest(state, pl["after"], pl["days"], today)
