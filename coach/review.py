"""Turning exam results into plan-change proposals that the learner accepts, edits or rejects."""
from __future__ import annotations

import datetime as dt
import uuid

import copy

from . import monitor
from . import plan as planlib

ADVANCE, RELEARN_BLOCKS, RELEARN_DAYS, INSERT_WEEK, CUT_SCOPE = (
    "advance",
    "relearn_blocks",
    "relearn_days",
    "insert_week",
    "cut_scope",
)


def _prop(entry_id, kind, title, detail, payload, today):
    return {
        "id": uuid.uuid4().hex[:8],
        "created": today.isoformat(),
        "source": entry_id,
        "type": kind,
        "title": title,
        "detail": detail,
        "payload": payload,
        "status": "pending",
    }


def _topics_text(topics):
    return ", ".join(topics[:5]) if topics else "the lowest-scoring topics"


def propose(state, attempt: dict, today: dt.date) -> list[dict]:
    plan = state["plan"]
    eid = attempt["entry"]
    total = attempt["total"]
    weak = attempt["weak_topics"]
    nxt = planlib.next_entry(plan, eid)
    cons = planlib.next_consolidation(plan, eid)
    prev = [a for a in state["attempts"] if a["id"] != attempt["id"]]
    prev_total = prev[-1]["total"] if prev else None
    out = []

    if total >= 0.8 and not (prev_total is not None and prev_total < 0.65 and total < 0.65):
        detail = "Score is 80% or higher. Keep the plan as it is."
        if weak:
            detail += f" Weak spots ({_topics_text(weak)}) are already on the review queue."
        out.append(_prop(eid, ADVANCE, "Advance as planned", detail, {}, today))
        return out

    two_low = prev_total is not None and prev_total < 0.65 and total < 0.65
    if total < 0.5 or two_low:
        reason = "below 50%" if total < 0.5 else "under 65% two weeks in a row"
        new_end = planlib.end_date(plan) + dt.timedelta(days=7)
        if new_end > planlib.max_end(plan):
            out.append(
                _prop(
                    eid,
                    CUT_SCOPE,
                    "Cut scope instead of adding a week",
                    f"Score is {reason}, but another week would end the plan after "
                    f"{planlib.max_end(plan):%d %b %Y}. Pick a later topic to shorten or drop on the Plan page, "
                    f"and relearn {_topics_text(weak)} in its place.",
                    {"topics": weak},
                    today,
                )
            )
        else:
            out.append(
                _prop(
                    eid,
                    INSERT_WEEK,
                    "Insert an extra consolidation week",
                    f"Score is {reason}. Add a 7-day relearning week right after {eid} for: {_topics_text(weak)}. "
                    f"Later dates move by one week (new end {new_end:%d %b %Y}).",
                    {"after": eid, "topics": weak},
                    today,
                )
            )
        return out

    if total < 0.65 and nxt:
        note = f"Mon–Tue: relearning protocol for {_topics_text(weak)} before new material."
        out.append(
            _prop(
                eid,
                RELEARN_DAYS,
                f"Turn the first two days of {nxt['id']} into relearning",
                note + (f" Also flag these topics for {cons['id']}." if cons else ""),
                {"entry": nxt["id"], "note": note, "topics": weak, "consolidation": cons["id"] if cons else None},
                today,
            )
        )
        return out

    if nxt:  # 65–79%
        note = f"Relearning blocks: Monday interleaving slot + Saturday stretch on {_topics_text(weak)}."
        out.append(
            _prop(
                eid,
                RELEARN_BLOCKS,
                f"Add relearning blocks to {nxt['id']}",
                note,
                {"entry": nxt["id"], "note": note, "topics": weak},
                today,
            )
        )
    return out


def apply(state, proposal: dict, today: dt.date, detail_override: str | None = None):
    """Apply an accepted proposal. The plan is snapshotted first so the change can be undone."""
    p = proposal
    before = copy.deepcopy(state["plan"])
    planlib.snapshot(state, f"{p['id']}: {p['title']}", today)
    try:
        _apply(state, p, today, detail_override)
        if planlib.end_date(state["plan"]) > planlib.max_end(state["plan"]):
            raise ValueError(f"This would end the plan after {planlib.max_end(state['plan']):%d %b %Y}. "
                             "Shorten or drop a later week first, or raise the latest end date in Settings.")
    except Exception:
        state["plan"] = before
        state["snapshots"].pop()
        raise
    p["status"] = "accepted"
    p["decided"] = today.isoformat()
    if detail_override:
        p["detail"] = detail_override


def undo(state, proposal: dict, today: dt.date):
    """Restore the plan as it was just before this proposal was accepted."""
    tag = f"{proposal['id']}:"
    for i in range(len(state.get("snapshots", [])) - 1, -1, -1):
        if state["snapshots"][i]["label"].startswith(tag):
            state["plan"] = copy.deepcopy(state["snapshots"][i]["plan"])
            del state["snapshots"][i:]
            proposal["status"] = "undone"
            planlib._log(state, "undo", f"{proposal['source']}: {proposal['title']}", today)
            return
    raise ValueError("No saved copy of the plan for that change (only the last 10 changes can be undone).")


def _apply(state, p: dict, today: dt.date, detail_override: str | None):
    note = (detail_override or "").strip() or p["payload"].get("note") or p["detail"]
    if p["type"] in monitor.MONITOR_TYPES:
        monitor.apply(state, p, today)
        if detail_override:
            planlib._log(state, "note", f"{p['source']}: {detail_override}", today)
    elif p["type"] in (RELEARN_BLOCKS, RELEARN_DAYS):
        planlib.add_note(state, p["payload"]["entry"], note, today)
        if p["type"] == RELEARN_DAYS and p["payload"].get("consolidation"):
            planlib.add_focus(state, p["payload"]["consolidation"], p["payload"]["topics"], today)
    elif p["type"] == INSERT_WEEK:
        new_id = planlib.insert_extra_week(state, p["payload"]["after"], p["payload"]["topics"], today)
        if detail_override:
            planlib.add_note(state, new_id, detail_override, today)
    elif p["type"] == CUT_SCOPE:
        planlib._log(state, "decision", f"Scope cut needed: {note}", today)


def reject(state, proposal: dict, today: dt.date):
    proposal["status"] = "rejected"
    proposal["decided"] = today.isoformat()
    planlib._log(state, "rejected", f"{proposal['source']}: {proposal['title']}", today)
