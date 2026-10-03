"""Spaced revision and relearning.

Key results (derivations or implementations to reproduce from a blank page) come back on an
expanding schedule (+1, +3, +7, +21, +60 days) and are self-checked on the Review queue page.
Due items, plus the concepts missed in the last exam, also steer the review questions that are
generated fresh for each exam (questions themselves are never stored).

A success moves an item one step further out. A failure sends it back to +1 day
and marks it for relearning.
"""
from __future__ import annotations

import datetime as dt

INTERVALS = [1, 3, 7, 21, 60]


def _iso(d: dt.date) -> str:
    return d.isoformat()


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def next_due(step: int, today: dt.date) -> dt.date:
    return today + dt.timedelta(days=INTERVALS[min(step, len(INTERVALS) - 1)])


# ---- key results -------------------------------------------------------------

def add_key_results(state, entry: dict, today: dt.date):
    for i, text in enumerate(entry.get("key_results", [])):
        rid = f"{entry['id']}:{i}"
        if rid in state["review_items"]:
            continue
        state["review_items"][rid] = {
            "id": rid,
            "entry": entry["id"],
            "text": text,
            "step": 0,
            "due": _iso(next_due(0, today)),
            "relearn": False,
            "history": [],
        }


def add_relearn_item(state, entry_id: str, topic: str, today: dt.date):
    rid = f"{entry_id}:relearn:{topic}"
    item = state["review_items"].get(rid)
    if item is None:
        state["review_items"][rid] = {
            "id": rid,
            "entry": entry_id,
            "text": f"Relearn: {topic}",
            "step": 0,
            "due": _iso(next_due(0, today)),
            "relearn": True,
            "history": [],
        }
    else:
        item.update(step=0, due=_iso(next_due(0, today)), relearn=True)


def due_items(state, today: dt.date) -> list[dict]:
    items = [i for i in state["review_items"].values() if _date(i["due"]) <= today]
    return sorted(items, key=lambda i: (not i["relearn"], i["due"]))


def all_history(state) -> list[dict]:
    """Every review event across all items, oldest first — the raw series behind the improvement chart."""
    events = []
    for it in state.get("review_items", {}).values():
        for h in it.get("history", []):
            events.append({"date": h["date"], "ok": bool(h["ok"]), "text": it["text"]})
    events.sort(key=lambda e: e["date"])
    return events


def mark_item(state, item_id: str, success: bool, today: dt.date):
    item = state["review_items"][item_id]
    item["history"].append({"date": _iso(today), "ok": bool(success)})
    if success:
        item["step"] = min(item["step"] + 1, len(INTERVALS) - 1)
        item["relearn"] = False
    else:
        item["step"] = 0
        item["relearn"] = True
    item["due"] = _iso(next_due(item["step"] if success else 0, today))
