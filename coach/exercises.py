"""Practice exercises and assignments.

Unlike exam questions (never stored, closed-book), these are explicitly meant to be kept and
reused: each week keeps a small cached pool so opening the page never costs an API call, and a
learner can also generate a brand new one on the spot when the cached ones feel stale.
"""
from __future__ import annotations

import datetime as dt
import uuid

from . import llm

EXERCISE_SYSTEM = (
    "You set short practice exercises and assignments for a self-taught deep-learning researcher who derives "
    "everything from first principles and builds every method from scratch. Each exercise is a concrete task "
    "to attempt on paper or in code, not a multiple-choice question — a derivation, a proof sketch, a small "
    "coding assignment, or a puzzle. Give a hint and a worked solution sketch, but never full working code: "
    "this is practice, not a graded submission. Use plain text and simple math notation (no LaTeX). "
    "Return JSON only."
)

EXERCISE_SCHEMA = (
    '{"kind": "derivation"|"proof"|"coding"|"puzzle", "prompt": str (the task, self-contained), '
    '"hint": str (one nudge, not the answer), "solution_sketch": str (3-6 sentences: the approach and the '
    'result, enough to self-check against, not a full solution), "difficulty": 1|2|3}'
)


def exercise_prompt(entry: dict, n: int, focus: list[str] | None = None) -> str:
    parts = [
        f"Week: {entry['id']} — {entry['title']}",
        "Math topics: " + "; ".join(entry.get("math", [])),
        "Build topics: " + "; ".join(entry.get("build", [])),
    ]
    if focus:
        parts.append("Concentrate on these topics: " + "; ".join(focus))
    parts.append(
        f"Write {n} NEW practice exercises for this week, each a different kind (derivation, proof, coding, "
        "puzzle) where it fits the topics, and vary difficulty 1-3."
    )
    parts.append('Return {"exercises": [' + EXERCISE_SCHEMA + ', ...]}')
    return "\n".join(parts)


def generate(settings, state, entry: dict, n: int = 3, focus=None, client=None, today=None) -> list[dict]:
    """Ask the model for n new exercises. Costs and logs one API call; the caller decides whether to cache them."""
    prompt = exercise_prompt(entry, n, focus)
    raw = llm.chat_json(settings, state, EXERCISE_SYSTEM, prompt, f"exercises {entry['id']}", client, today)
    items = raw.get("exercises") if isinstance(raw, dict) else None
    out = []
    if isinstance(items, list):
        for e in items:
            try:
                out.append({
                    "id": uuid.uuid4().hex[:8],
                    "entry": entry["id"],
                    "kind": str(e.get("kind", "exercise")).strip().lower() or "exercise",
                    "prompt": str(e["prompt"]).strip(),
                    "hint": str(e.get("hint", "")).strip(),
                    "solution_sketch": str(e.get("solution_sketch", "")).strip(),
                    "difficulty": int(e.get("difficulty", 2)),
                    "created": (today or dt.date.today()).isoformat(),
                    "done": False,
                })
            except (KeyError, TypeError, ValueError):
                continue
    return out


def cached(state, entry_id: str) -> list[dict]:
    return state.get("exercises", {}).get(entry_id, [])


def add_to_cache(state, entry_id: str, new_items: list[dict]):
    state.setdefault("exercises", {}).setdefault(entry_id, []).extend(new_items)


def mark_done(state, entry_id: str, exercise_id: str, done: bool):
    for e in cached(state, entry_id):
        if e["id"] == exercise_id:
            e["done"] = done
            return
