"""The only module that talks to OpenAI, with a hard monthly spending cap.

Every call is priced from the token counts the API returns and logged in the
state. Before a call, the worst-case cost is estimated; if it would push the
month over the budget, the call is refused instead of made.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import uuid
from typing import Any

MAX_OUTPUT_TOKENS = 8000


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


def month_key(day: dt.date | None = None) -> str:
    return (day or dt.date.today()).strftime("%Y-%m")


def spent_this_month(state, day: dt.date | None = None) -> float:
    m = month_key(day)
    return round(sum(c["usd"] for c in state["cost_log"] if c["date"].startswith(m)), 6)


def cost_usd(settings, in_tokens: int, out_tokens: int) -> float:
    return in_tokens / 1e6 * settings.price_input_per_m + out_tokens / 1e6 * settings.price_output_per_m


def _client(settings):
    import os

    if os.environ.get("COACH_FAKE_LLM") == "1":  # offline demos and tests only
        from .fake_llm import FakeClient

        return FakeClient()
    if not settings.openai_api_key:
        raise LLMError("No OpenAI key set. Add OPENAI_API_KEY to the app's secrets.")
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def chat_json(settings, state, system: str, user: str, purpose: str, client=None, today=None) -> dict:
    """One JSON-mode chat call, budget-checked and logged. Retries once on bad JSON."""
    est_in = (len(system) + len(user)) // 3 + 50
    worst = cost_usd(settings, est_in, MAX_OUTPUT_TOKENS)
    spent = spent_this_month(state, today)
    if spent + worst > settings.monthly_budget_usd:
        raise BudgetExceeded(
            f"This call could cost up to ${worst:.4f}; ${spent:.4f} of the ${settings.monthly_budget_usd:.2f} "
            "monthly budget is already used. Raise MONTHLY_BUDGET_USD or wait for next month."
        )
    client = client or _client(settings)
    kwargs: dict[str, Any] = dict(
        model=settings.openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    if settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    last_err = None
    for _ in range(2):
        resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        state["cost_log"].append(
            {
                "date": (today or dt.date.today()).isoformat(),
                "purpose": purpose,
                "model": settings.openai_model,
                "in_tokens": tin,
                "out_tokens": tout,
                "usd": round(cost_usd(settings, tin, tout), 6),
            }
        )
        text = resp.choices[0].message.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
    raise LLMError(f"The model did not return valid JSON twice in a row: {last_err}")


# ---- multiple-choice generation ---------------------------------------------

MCQ_SYSTEM = (
    "You write rigorous exams for a self-taught deep-learning researcher who derives everything from first "
    "principles. Questions must test understanding and implementation, not trivia or memorised names. Each "
    "question has exactly 4 options; an \"mcq\" or \"puzzle\" question has exactly one correct option, an "
    "\"msq\" question has 1 to 3. Distractors must be plausible mistakes (sign errors, wrong shapes, confused "
    "concepts) — an msq's wrong options must each be wrong for a different reason, not variations on one slip. "
    "Use plain text and simple math notation (no LaTeX). Return JSON only."
)


QUESTION_SCHEMA = (
    '{"type": "mcq"|"msq"|"puzzle", "question": str, "options": [str, str, str, str], '
    '"answer": 0-3 for mcq/puzzle, or a list of 1-3 ints (0-3) for msq — every correct option, '
    '"explanation": str (2-4 sentences: why the answer is right and why the most tempting wrong option is wrong), '
    '"concept": str (one short sentence naming the exact idea tested), '
    '"topic": short topic name, "difficulty": 1|2|3}'
)


def mcq_prompt(entry: dict, n: int, focus: list[str] | None = None,
               review: list[str] | None = None, n_review: int = 0, allow_msq: bool = True) -> str:
    """One prompt that asks for the week's questions and, optionally, review questions."""
    parts = [
        f"Week: {entry['id']} — {entry['title']}",
        "Math topics: " + "; ".join(entry.get("math", [])),
        "Build topics: " + "; ".join(entry.get("build", [])),
    ]
    if focus:
        parts.append("Concentrate on these weak topics: " + "; ".join(focus))
    parts.append(
        f"Write {n} NEW questions for this week: about a third conceptual 'why' questions (type \"mcq\", and make "
        "one or two of these a type \"puzzle\" — a short brain-teaser or thought experiment rather than a plain "
        "question), a third short derivations or calculations (numbers the reader can check by hand), and a third "
        "about implementation (reading or predicting code, tensor shapes, numerical stability, debugging). "
        + (f"Make two or three of the {n} type \"msq\" (multi-select, 1-3 correct options) instead of \"mcq\" — "
           "pick ones where several options are genuinely correct or genuinely wrong, never a single-answer "
           "question dressed up as multi-select. " if allow_msq else "Every question is type \"mcq\" or \"puzzle\" "
           "(single correct option) — no \"msq\". ")
        + "Vary difficulty 1-3."
    )
    if review and n_review:
        parts.append(
            f"Also write {n_review} REVIEW questions. They must re-test these concepts the learner missed in the "
            "last exam or has due for revision, from a different angle than a plain restatement (a new example, "
            "a calculation, or a code variant):\n- " + "\n- ".join(review[:20])
        )
    parts.append(
        'Return {"questions": [' + QUESTION_SCHEMA + ', ...]'
        + (', "review": [same fields, ...]' if review and n_review else "")
        + "}"
    )
    return "\n".join(parts)


def validate_mcqs(raw, entry_id: str, seed: int = 0, key: str = "questions", minimum: int = 4) -> list[dict]:
    qs = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(qs, list):
        if minimum == 0:
            return []
        raise LLMError(f"Model reply had no '{key}' list.")
    rng = random.Random(seed)
    out = []
    for q in qs:
        try:
            opts = [str(o) for o in q["options"]]
            if len(opts) != 4 or len(set(opts)) != 4:
                continue
            qtype = str(q.get("type", "mcq")).strip().lower()
            if qtype not in ("mcq", "msq", "puzzle"):
                qtype = "mcq"
            raw_ans = q["answer"]
            if qtype == "msq":
                ans_set = sorted({int(a) for a in raw_ans}) if isinstance(raw_ans, list) else [int(raw_ans)]
                if not (1 <= len(ans_set) <= 3) or not all(0 <= a < 4 for a in ans_set):
                    continue
            else:
                a0 = raw_ans[0] if isinstance(raw_ans, list) else raw_ans
                ans_set = [int(a0)]
                if not 0 <= ans_set[0] < 4:
                    continue
            # Shuffle so the correct answer isn't always in the model's favourite slot.
            order = list(range(4))
            rng.shuffle(order)
            new_ans = sorted(order.index(a) for a in ans_set)
            out.append(
                {
                    "id": f"{entry_id}-{uuid.uuid4().hex[:8]}",
                    "entry": entry_id,
                    "type": qtype,
                    "question": str(q["question"]).strip(),
                    "options": [opts[i] for i in order],
                    "answer": new_ans if qtype == "msq" else new_ans[0],
                    "explanation": str(q.get("explanation", "")).strip(),
                    "concept": str(q.get("concept", "")).strip()[:200],
                    "topic": str(q.get("topic", "general")).strip() or "general",
                    "difficulty": int(q.get("difficulty", 2)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if len(out) < minimum:
        raise LLMError("Too few usable questions came back; try again.")
    return out


def generate_questions(settings, state, entry: dict, n: int = 12, focus=None, review=None, n_review: int = 0,
                       client=None, today=None, allow_msq: bool = True):
    """Fresh questions for one exam, never cached. Returns (week_questions, review_questions)."""
    prompt = mcq_prompt(entry, n, focus, review, n_review, allow_msq)
    raw = chat_json(settings, state, MCQ_SYSTEM, prompt, f"exam {entry['id']}", client, today)
    seed = len(state["cost_log"])
    week = validate_mcqs(raw, entry["id"], seed, minimum=min(4, n))
    rev = validate_mcqs(raw, entry["id"], seed + 1, key="review", minimum=0)[:n_review] if review and n_review else []
    return week, rev


def generate_mcqs(settings, state, entry: dict, n: int = 12, focus=None, client=None, today=None,
                  allow_msq: bool = True) -> list[dict]:
    return generate_questions(settings, state, entry, n, focus, client=client, today=today, allow_msq=allow_msq)[0]


# ---- short AI feedback after an exam ------------------------------------------

FEEDBACK_SYSTEM = (
    "You are a demanding but fair study coach. Given exam results, write brief, specific feedback. "
    "Never invent results. Return JSON only."
)


def feedback(settings, state, attempt: dict, entry: dict, client=None, today=None) -> dict:
    missed = [
        {"topic": m["topic"], "concept": m.get("concept") or m.get("question", "")[:200]}
        for m in attempt["mcq"]["items"] + attempt["spaced"]["items"]
        if not m["correct"]
    ][:12]
    payload = {
        "week": f"{entry['id']} {entry['title']}",
        "scores": {k: attempt[k]["score"] for k in ("mcq", "coding", "spaced") if attempt[k]["n"]},
        "total": attempt["total"],
        "topic_scores": attempt["by_topic"],
        "missed": missed,
        "coding": [{"task": c["title"], "score": c["score"]} for c in attempt["coding"]["items"]],
    }
    user = (
        json.dumps(payload)
        + '\nReturn {"strengths": [<=3 short items], "gaps": [<=4 items naming the exact concept to relearn], '
        '"next_week_focus": [<=3 concrete actions, e.g. which derivation to redo or which module to re-implement]}'
    )
    return chat_json(settings, state, FEEDBACK_SYSTEM, user, f"feedback {entry['id']}", client, today)
