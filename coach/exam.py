"""Building an exam for a plan entry, and scoring a submitted exam."""
from __future__ import annotations

import datetime as dt
import uuid
from collections import defaultdict

from . import coding, llm, spaced
from . import plan as planlib

WEIGHTS = {"mcq": 0.4, "coding": 0.4, "spaced": 0.2}
MCQ_COUNT = 12


def _earlier_ids(plan, entry_id):
    i = planlib.index_of(plan, entry_id)
    return [e["id"] for e in plan["entries"][:i]]


def weak_topics_since(state, entry_ids: list[str], threshold=0.7) -> list[str]:
    agg = defaultdict(list)
    for a in state["attempts"]:
        if a["entry"] in entry_ids:
            for t, s in a["by_topic"].items():
                agg[t].append(s)
    weak = [(sum(v) / len(v), t) for t, v in agg.items() if sum(v) / len(v) < threshold]
    return [t for _, t in sorted(weak)][:8]


def _coding_ids_for(state, entry) -> list[str]:
    fixed = list(entry.get("tasks", []))
    if entry["kind"] in ("topic", "tooling"):
        return fixed
    # Consolidation / extra / capstone: top up with the weakest (or never-tried) earlier tasks.
    plan = state["plan"]
    earlier = []
    pool = entry.get("covers") if entry["kind"] == "depth" and entry.get("covers") else _earlier_ids(plan, entry["id"])
    for eid in pool:
        earlier.extend(planlib.find(plan, eid).get("tasks", []))
    best = {}
    for a in state["attempts"]:
        for c in a["coding"]["items"]:
            best[c["task_id"]] = max(best.get(c["task_id"], 0.0), c["score"])
    ranked = sorted(
        [t for t in earlier if t not in fixed], key=lambda t: (best.get(t, -1.0) >= 1.0, best.get(t, -1.0))
    )
    return fixed + ranked[: max(0, 2 - len(fixed))]


def review_concepts(state, today: dt.date) -> list[str]:
    """What the review part should re-test: concepts missed last exam, then due and relearning items."""
    out = []
    last = state.get("last_exam") or {}
    out.extend(last.get("missed", []))
    for it in spaced.due_items(state, today):
        out.append(it["text"])
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:20]


def build_exam(settings, state, entry_id: str, today: dt.date, client=None) -> dict:
    """A fresh exam every time. Questions live only in the browser session and are never saved."""
    plan = state["plan"]
    entry = planlib.find(plan, entry_id)
    if entry["kind"] in planlib.NO_EXAM:
        raise ValueError(f"{entry_id} is a rest week: no exam.")
    focus = list(entry.get("focus", []))
    if entry["kind"] == "depth":
        focus = focus or weak_topics_since(state, entry.get("covers", []))
    if entry["kind"] in ("consolidation", "extra"):
        phase_ids = [e["id"] for e in plan["entries"] if e.get("phase") == entry.get("phase") and e["id"] != entry_id]
        focus = focus or weak_topics_since(state, phase_ids)
    elif entry["kind"] == "capstone":
        focus = focus or weak_topics_since(state, _earlier_ids(plan, entry_id))
    concepts = review_concepts(state, today)
    cap = 12 if entry["kind"] in ("consolidation", "extra", "capstone") else 8
    n_review = min(cap, 2 * len(concepts))
    mcq, review_qs = llm.generate_questions(
        settings, state, entry, MCQ_COUNT, focus or None, concepts, n_review, client, today
    )
    tasks = [coding.load_task(t) for t in _coding_ids_for(state, entry)]
    return {
        "id": uuid.uuid4().hex[:10],
        "entry": entry_id,
        "mcq": mcq,
        "tasks": [t.__dict__ for t in tasks],
        "spaced": review_qs,
    }


PLACEMENT_PASS = 0.7
PLACEMENT_MAX_WEEKS = 12
PLACEMENT_Q_PER_WEEK = 2


def build_placement_exam(settings, state, target_id: str, today: dt.date, client=None) -> dict:
    """A short check spanning every week between here and target_id. Pass it to skip straight there."""
    skipped = planlib.entries_between(state["plan"], target_id)
    if not skipped:
        raise ValueError("There's nothing to skip to reach that week.")
    if len(skipped) > PLACEMENT_MAX_WEEKS:
        raise ValueError(f"That's {len(skipped)} weeks away — placement checks cover at most {PLACEMENT_MAX_WEEKS} at once. Pick a nearer week.")
    items = []
    for e in skipped:
        qs = llm.generate_mcqs(settings, state, e, PLACEMENT_Q_PER_WEEK, client=client, today=today, allow_msq=False)
        for q in qs:
            q["source_entry"] = e["id"]
        items.extend(qs)
    return {"id": uuid.uuid4().hex[:10], "target": target_id, "skipped": [e["id"] for e in skipped], "mcq": items}


def grade_placement(exam: dict, answers: dict, today: dt.date) -> dict:
    items = []
    for q in exam["mcq"]:
        chosen = answers.get(q["id"])
        items.append({**q, "chosen": chosen, "correct": chosen == q["answer"]})
    by_entry = defaultdict(list)
    for i in items:
        by_entry[i["source_entry"]].append(1.0 if i["correct"] else 0.0)
    by_entry_score = {e: round(sum(v) / len(v), 3) for e, v in by_entry.items()}
    total = round(sum(i["correct"] for i in items) / len(items), 4) if items else 0.0
    weakest = sorted(by_entry_score, key=lambda e: by_entry_score[e])
    return {
        "id": exam["id"], "target": exam["target"], "skipped": exam["skipped"], "date": today.isoformat(),
        "items": items, "total": total, "by_entry": by_entry_score,
        "passed": total >= PLACEMENT_PASS, "weakest": weakest[:3],
    }


def _grade_mcqs(questions, answers):
    items = []
    for q in questions:
        chosen = answers.get(q["id"])
        qtype = q.get("type", "mcq")
        if qtype == "msq":
            correct = sorted(chosen or []) == sorted(q["answer"])
        else:
            correct = chosen == q["answer"]
        items.append(
            {
                "qid": q["id"],
                "type": qtype,
                "topic": q["topic"],
                "concept": q.get("concept", ""),
                "question": q["question"],
                "options": q["options"],
                "chosen": chosen,
                "answer": q["answer"],
                "correct": correct,
                "explanation": q.get("explanation", ""),
            }
        )
    n = len(items)
    return {"items": items, "n": n, "score": (sum(i["correct"] for i in items) / n) if n else 0.0}


def grade(exam: dict, answers: dict, code: dict, today: dt.date, run=coding.run_tests) -> dict:
    mcq = _grade_mcqs(exam["mcq"], answers)
    spaced_part = _grade_mcqs(exam["spaced"], answers)
    c_items = []
    for t in exam["tasks"]:
        src = code.get(t["id"], t["starter"])
        r = run(src, t["tests"])
        c_items.append(
            {
                "task_id": t["id"],
                "title": t["title"],
                "score": r.score,
                "passed": r.passed,
                "total": r.total,
                "failures": r.failures,
                "output": r.output[-3000:],
                "code": src,
            }
        )
    n_c = len(c_items)
    coding_part = {"items": c_items, "n": n_c, "score": (sum(c["score"] for c in c_items) / n_c) if n_c else 0.0}

    parts = {"mcq": mcq, "coding": coding_part, "spaced": spaced_part}
    w = {k: WEIGHTS[k] for k, p in parts.items() if p["n"]}
    total = sum(parts[k]["score"] * w[k] for k in w) / sum(w.values()) if w else 0.0

    by = defaultdict(list)
    for i in mcq["items"] + spaced_part["items"]:
        by[i["topic"]].append(1.0 if i["correct"] else 0.0)
    for c in c_items:
        by[c["title"]].append(c["score"])
    by_topic = {t: round(sum(v) / len(v), 3) for t, v in by.items()}
    counts = {t: len(v) for t, v in by.items()}
    weak = sorted(
        [t for t, s in by_topic.items() if (s < 0.5 and counts[t] >= 2) or (t in {c["title"] for c in c_items} and s < 0.7)],
        key=lambda t: by_topic[t],
    )
    return {
        "id": exam["id"],
        "entry": exam["entry"],
        "date": today.isoformat(),
        "mcq": mcq,
        "coding": coding_part,
        "spaced": spaced_part,
        "total": round(total, 4),
        "by_topic": by_topic,
        "weak_topics": weak,
    }


def _slim(attempt: dict) -> dict:
    """What is kept after an exam: scores and concepts, never the question text or options."""
    keep_q = lambda i: {"topic": i["topic"], "concept": i.get("concept", ""), "correct": i["correct"]}
    slim = {k: attempt[k] for k in ("id", "entry", "date", "total", "by_topic", "weak_topics")}
    for part in ("mcq", "spaced"):
        p = attempt[part]
        slim[part] = {"n": p["n"], "score": p["score"], "items": [keep_q(i) for i in p["items"]]}
    c = attempt["coding"]
    slim["coding"] = {"n": c["n"], "score": c["score"], "items": [
        {k: i[k] for k in ("task_id", "title", "score", "passed", "total", "failures")} for i in c["items"]]}
    return slim


def record(state, attempt: dict, today: dt.date):
    """Store a slim copy of the graded attempt, remember what was missed, update the plan."""
    state["attempts"].append(_slim(attempt))
    missed = []
    for part in ("mcq", "spaced"):
        for i in attempt[part]["items"]:
            if not i["correct"]:
                missed.append(f"{i['topic']}: {i.get('concept') or 'see topic'}")
    for c in attempt["coding"]["items"]:
        if c["score"] < 1.0:
            missed.append(f"{c['title']} (coding): failed {', '.join(c['failures'][:3]) or 'tests'}")
    state["last_exam"] = {"entry": attempt["entry"], "date": attempt["date"],
                          "missed": missed[:20], "weak_topics": attempt["weak_topics"]}
    entry = planlib.find(state["plan"], attempt["entry"])
    spaced.add_key_results(state, entry, today)
    for t in attempt["weak_topics"]:
        spaced.add_relearn_item(state, entry["id"], t, today)
    entry["status"] = "done"
    entry["score"] = attempt["total"]
