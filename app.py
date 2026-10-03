"""DL Coach — weekly exams, spaced revision and an adaptive study plan.

Run locally:   streamlit run app.py
Deploy:        see README.md
"""
from __future__ import annotations

import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from coach import auth, config, exercises as exlib, llm, monitor, review, spaced, storage

DAILY_TIPS = [
    "Before opening your notes, try to restate yesterday's key result from memory first.",
    "Explain today's hardest idea out loud as if to someone who has never seen it.",
    "Re-derive one proof from this week on a blank page before checking it against your notes.",
    "Do one due review item before starting anything new today.",
    "Write down the one thing that confused you today — it's tomorrow's review question.",
    "Time-box a stuck problem to 20 minutes, then move on and come back to it later.",
    "Re-read code you wrote two weeks ago — does it still make sense without re-deriving it?",
]
from coach import exam as examlib
from coach import plan as planlib

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PLAN = os.path.join(ROOT, "data", "plan.yaml")

st.set_page_config(page_title="DL Coach", layout="wide")

settings = config.load_settings()
email = auth.require_login(settings)
today = dt.datetime.now(ZoneInfo(settings.timezone)).date()
ss = st.session_state


# ---- state ---------------------------------------------------------------------

def store():
    if "store" not in ss:
        ss.store = storage.make_store(settings)
    return ss.store


def state():
    if "state" not in ss:
        try:
            ss.state = store().load()
        except storage.StorageError as e:
            st.error(f"Couldn't load your saved progress: {e}")
            st.stop()
        if not ss.state.get("plan"):
            planlib.ensure_plan(ss.state, BASE_PLAN)
            save("Create plan")
    return ss.state


def save(message: str):
    try:
        store().save(ss.state, message)
        return True
    except storage.StorageError as e:
        st.error(f"Saving failed: {e}. Your changes are kept in this session; try again from Settings.")
        return False


S = state()
P = S["plan"]


def fmt_range(e):
    return f"{e['start']:%d %b} – {e['end']:%d %b %Y}"


def pct(x):
    return f"{100 * x:.0f}%"


# ---- sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("### DL Coach")
    st.caption(f"Signed in as {email}")
    page = st.radio(
        "Go to",
        ["Dashboard", "Weekly exam", "Exercises", "Review & plan", "Review queue", "Settings & cost"],
        label_visibility="collapsed",
        key="nav",
    )
    spent = llm.spent_this_month(S, today)
    st.progress(min(1.0, spent / max(settings.monthly_budget_usd, 1e-9)), text=f"API this month: ${spent:.3f} of ${settings.monthly_budget_usd:.2f}")
    due_n = len(spaced.due_items(S, today))
    pend_n = sum(p["status"] == "pending" for p in S["proposals"])
    if pend_n:
        st.warning(f"{pend_n} plan change{'s' if pend_n != 1 else ''} waiting for your approval (Review & plan)")
    if due_n:
        st.info(f"{due_n} review item{'s' if due_n != 1 else ''} due today")
    try:
        if "auth" in st.secrets:
            st.button("Sign out", on_click=st.logout)
    except Exception:
        pass


# ---- pages ---------------------------------------------------------------------

def page_dashboard():
    sched = planlib.schedule(P)
    cur = planlib.current_entry(P)
    by_date = planlib.entry_on(P, today)
    st.title("Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    done = sum(e["status"] == "done" for e in sched)
    c1.metric("Weeks done", f"{done} / {len(sched)}")
    scores = [a["total"] for a in S["attempts"]]
    c2.metric("Last exam", pct(scores[-1]) if scores else "—",
              delta=(f"{100 * (scores[-1] - scores[-2]):+.0f} pts" if len(scores) > 1 else None))
    end = planlib.end_date(P)
    c3.metric("Plan ends", f"{end:%d %b %Y}", help=f"Latest allowed end: {planlib.max_end(P):%d %b %Y}")
    c4.metric("Review due", due_n)

    # Progress monitoring: proposals only — nothing changes until you accept on Review & plan.
    if ss.get("monitored_on") != today.isoformat():
        new = monitor.check(S, today)
        ss.monitored_on = today.isoformat()
        if new:
            save(f"Monitor: {len(new)} proposal(s)")
    pend = [p for p in S["proposals"] if p["status"] == "pending"]
    if pend:
        st.warning(f"**{len(pend)} proposed change{'s' if len(pend) != 1 else ''} to your plan** — "
                   + "; ".join(p["title"] for p in pend[:3]) + ". Nothing changes until you accept it on **Review & plan**.")

    streak = monitor.study_streak(S, today)
    m1, m2 = st.columns([1, 3])
    m1.metric("Study streak", f"{streak} day{'s' if streak != 1 else ''}")
    m2.info("Today's idea: " + DAILY_TIPS[today.toordinal() % len(DAILY_TIPS)])

    week = by_date or cur
    with st.container(border=True):
        st.subheader(f"This week: {week['id']} — {week['title']}")
        st.caption(fmt_range(week) + f" · {week['kind']}")
        if week["kind"] == "rest":
            st.success("Rest week. No study, no exam. Optional: up to 30 minutes a day on due review items.")
        else:
            a, b, c = st.columns(3)
            planned = monitor.planned_hours(S, week, today)
            logged = monitor.hours_logged(S, week["start"], min(today, week["end"]))
            a.metric("Hours logged this week", f"{logged:.1f}", help=f"Planned so far: {planned:.0f} h (edit daily targets in Settings)")
            b.metric("Checklist", pct(monitor.completion(S, week)))
            c.metric("Blog posts", sum(1 for v in S["blog"].values() if v))
            l1, l2, l3 = st.columns([1, 3, 1])
            prev = next((r for r in S["daily_log"] if r["date"] == today.isoformat()), None)
            hrs = l1.number_input("Hours today", 0.0, 16.0, float(prev["hours"]) if prev else 0.0, 0.5, key="log_h")
            note = l2.text_input("What you did / what was hard", prev["note"] if prev else "", key="log_n")
            if l3.button("Log today", type="primary"):
                S["daily_log"] = [r for r in S["daily_log"] if r["date"] != today.isoformat()]
                S["daily_log"].append({"date": today.isoformat(), "entry": week["id"], "hours": hrs, "note": note})
                save(f"Log {today}: {hrs} h"); st.rerun()
            its = monitor.items(week)
            if its:
                with st.expander("Checklist for this week", expanded=False):
                    ticked = set(S["checklist"].get(week["id"], []))
                    chosen = [i for i in its if st.checkbox(i, value=i in ticked, key=f"ck_{week['id']}_{its.index(i)}")]
                    if set(chosen) != ticked and st.button("Save checklist"):
                        S["checklist"][week["id"]] = chosen
                        save(f"Checklist {week['id']}"); st.rerun()
                sig = monitor.pace_signal(S, week, today)
                if sig:
                    st.caption(f"Recomputed from what you've logged yourself — {sig['why']}.")
                    if sig["behind"]:
                        st.warning("Behind pace at today's rate.")
                        if st.button("Propose an adjustment now", key=f"pace_{week['id']}"):
                            new = monitor.propose_now(S, week, today)
                            save(f"Pace proposal: {len(new)}")
                            if new:
                                st.toast(f"{len(new)} proposal(s) added — review and accept on Review & plan.")
                            else:
                                st.toast("Already proposed recently — check Review & plan.")
                            st.rerun()
            url = st.text_input("This week's blog post (URL)", S["blog"].get(week["id"], ""), key=f"blog_{week['id']}")
            if url != S["blog"].get(week["id"], "") and st.button("Save blog link"):
                S["blog"][week["id"]] = url.strip()
                save(f"Blog {week['id']}"); st.rerun()

    st.subheader(f"Next exam: {cur['id']} — {cur['title']}")
    st.caption(fmt_range(cur) + (f" · today is in {by_date['id']}" if by_date and by_date["id"] != cur["id"] else ""))
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Math track**")
        st.markdown("\n".join(f"- {m}" for m in cur["math"]) or "—")
    with cols[1]:
        st.markdown("**Build track**")
        st.markdown("\n".join(f"- {b}" for b in cur["build"]) or "—")
    if cur.get("notes") or cur.get("focus"):
        st.warning("**Adjustments from your reviews**\n\n" + "\n".join(f"- {n}" for n in cur.get("notes", []) + [f"Relearn: {f}" for f in cur.get("focus", [])]))

    if S["attempts"]:
        st.subheader("Scores")
        df = pd.DataFrame(
            [{"exam": f"{a['entry']} ({a['date']})", "total": 100 * a["total"],
              "multiple choice": 100 * a["mcq"]["score"] if a["mcq"]["n"] else None,
              "coding": 100 * a["coding"]["score"] if a["coding"]["n"] else None,
              "review": 100 * a["spaced"]["score"] if a["spaced"]["n"] else None} for a in S["attempts"]]
        ).set_index("exam")
        st.line_chart(df, y_label="score (%)")

    hist = spaced.all_history(S)
    if hist:
        st.subheader("Revision improvement")
        window = min(8, len(hist))
        roll = pd.Series([100.0 if h["ok"] else 0.0 for h in hist]).rolling(window, min_periods=1).mean()
        df2 = pd.DataFrame({"review #": range(1, len(hist) + 1), "rolling success rate": roll.values}).set_index("review #")
        st.line_chart(df2, y_label="success rate (%)")
        overall = 100 * sum(h["ok"] for h in hist) / len(hist)
        st.caption(f"{sum(h['ok'] for h in hist)} of {len(hist)} spaced-review attempts correct overall ({overall:.0f}%), "
                   f"rolling over the last {window}.")

    st.subheader("Plan")
    def _topics(e):
        ts = e.get("math", []) + e.get("build", [])
        text = "; ".join(ts[:3])
        return text + ("…" if len(ts) > 3 else "") if ts else ""
    rows = [{"week": e["id"], "dates": fmt_range(e), "title": e["title"], "topics": _topics(e), "kind": e["kind"],
             "status": "done" if e["status"] == "done" else ("now" if e["id"] == cur["id"] else ""),
             "score": pct(e["score"]) if e.get("score") is not None else ""} for e in sched]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _show_results(attempt):
    st.subheader(f"Result: {pct(attempt['total'])}")
    cols = st.columns(3)
    for col, (k, label) in zip(cols, [("mcq", "Multiple choice"), ("coding", "Coding"), ("spaced", "Review")]):
        p = attempt[k]
        col.metric(label, pct(p["score"]) if p["n"] else "—", help=f"{p['n']} items")
    if attempt["weak_topics"]:
        st.warning("Weak topics (added to your review queue for relearning): " + ", ".join(attempt["weak_topics"]))

    with st.expander("Multiple-choice answers", expanded=False):
        for part in ("mcq", "spaced"):
            for i, it in enumerate(attempt[part]["items"], 1):
                mark = "✅" if it["correct"] else "❌"
                st.markdown(f"**{mark} {it['question']}**  \n_topic: {it['topic']}_")
                opts = it["options"]
                if it.get("type") == "msq":
                    chosen = ", ".join(opts[i] for i in it["chosen"]) if it["chosen"] else "no answer"
                    correct = ", ".join(opts[i] for i in it["answer"])
                else:
                    chosen = opts[it["chosen"]] if it["chosen"] is not None else "no answer"
                    correct = opts[it["answer"]]
                st.markdown(f"Your answer: {chosen}  \nCorrect: {correct}")
                if it["explanation"]:
                    st.caption(it["explanation"])
    for c in attempt["coding"]["items"]:
        with st.expander(f"Coding — {c['title']}: {c['passed']}/{c['total']} tests"):
            if c["failures"]:
                st.markdown("Failed: " + ", ".join(f"`{f}`" for f in c["failures"]))
                st.code(c["output"][-2000:], language="text")
            try:
                from coach.coding import load_task
                st.markdown("**Reference solution** — compare it with yours line by line.")
                st.code(load_task(c["task_id"]).solution, language="python")
            except Exception:
                pass

    fb = S["ai_feedback"].get(attempt["id"])
    if fb:
        st.markdown("**Coach feedback**")
        for k, label in [("strengths", "Strengths"), ("gaps", "Gaps"), ("next_week_focus", "Next week")]:
            if fb.get(k):
                st.markdown(f"*{label}:* " + "; ".join(map(str, fb[k])))
    elif st.button("Get written feedback (about $0.001)", key=f"fb_{attempt['id']}"):
        try:
            with st.spinner("Writing feedback…"):
                S["ai_feedback"][attempt["id"]] = llm.feedback(settings, S, attempt, planlib.find(P, attempt["entry"]), today=today)
            save(f"Feedback for {attempt['entry']}")
            st.rerun()
        except (llm.BudgetExceeded, llm.LLMError) as e:
            st.error(str(e))
    st.info("Plan changes based on this exam are waiting on the **Review & plan** page.")


def page_exam():
    st.title("Weekly exam")
    ids = planlib.exam_ids(P)
    cur = planlib.current_entry(P)
    if ss.get("exam_week") not in ids:
        ss.exam_week = cur["id"]
    eid = st.selectbox("Week", ids, key="exam_week",
                       format_func=lambda i: f"{i} — {planlib.find(P, i)['title']}" + (" (done)" if planlib.find(P, i)["status"] == "done" else ""))
    entry = planlib.find(P, eid)
    st.caption("Multiple choice 40% · coding 40% · review 20% (weights rescale when a part is empty). "
               "Closed-book: no notes, no AI, no internet.")

    ex = ss.get("exam")
    if ex and ex["entry"] != eid:
        ex = None
    last = ss.get("last_attempt")
    if last and last["entry"] == eid and not ex:
        _show_results(last)
        if st.button("Take this exam again"):
            ss.last_attempt = None
            st.rerun()
        return

    if not ex:
        concepts = examlib.review_concepts(S, today)
        st.write("Each exam is generated fresh by your OpenAI model (about $0.003) and the questions are never saved. "
                 + (f"The review part re-tests {len(concepts)} concept(s) from your last exam and your due revision items."
                    if concepts else "There is nothing to review yet, so this exam has no review part."))
        if concepts:
            with st.expander("Concepts the review part will cover"):
                st.markdown("\n".join(f"- {c}" for c in concepts))
        if st.button("Start exam", type="primary"):
            try:
                with st.spinner("Writing your exam…"):
                    ss.exam = examlib.build_exam(settings, S, eid, today)
                save(f"Exam generated for {eid} (cost logged)")
                st.rerun()
            except llm.BudgetExceeded as e:
                st.error(str(e))
            except llm.LLMError as e:
                st.error(f"Couldn't create questions: {e}")
            except Exception as e:  # network errors from the API client
                st.error(f"The OpenAI request failed: {e}")
        return

    def _question_widget(label_n, q, key):
        tag = {"msq": "**[Select all that apply]** ", "puzzle": "**[Puzzle]** "}.get(q.get("type", "mcq"), "")
        if q.get("type") == "msq":
            return st.multiselect(f"**{label_n}.** {tag}{q['question']}", range(4),
                                  format_func=lambda i, q=q: q["options"][i], key=key)
        return st.radio(f"**{label_n}.** {tag}{q['question']}", range(4), index=None,
                        format_func=lambda i, q=q: q["options"][i], key=key)

    with st.form("exam_form"):
        st.subheader(f"{entry['id']} — {entry['title']}")
        st.caption("Questions exist only in this browser tab. Don't refresh or close it until you submit.")
        answers = {}
        if ex["mcq"]:
            st.markdown("### Part 1 — Multiple choice")
            for n, q in enumerate(ex["mcq"], 1):
                answers[q["id"]] = _question_widget(n, q, f"a_{ex['id']}_{q['id']}")
        code = {}
        if ex["tasks"]:
            st.markdown("### Part 2 — Coding (hidden tests run when you submit)")
            for t in ex["tasks"]:
                st.markdown(t["prompt"])
                code[t["id"]] = st.text_area("Your code", value=t["starter"], height=360, key=f"c_{ex['id']}_{t['id']}")
        if ex["spaced"]:
            st.markdown("### Part 3 — Review (concepts you missed last time, and due revision)")
            for n, q in enumerate(ex["spaced"], 1):
                answers[q["id"]] = _question_widget(f"R{n}", q, f"s_{ex['id']}_{q['id']}")
        submitted = st.form_submit_button("Submit exam", type="primary")

    if st.button("Cancel this attempt"):
        ss.exam = None
        st.rerun()

    if submitted:
        unanswered = sum(v is None or v == [] for v in answers.values())
        with st.spinner("Running tests and scoring…"):
            attempt = examlib.grade(ex, answers, code, today)
            examlib.record(S, attempt, today)
            S["proposals"].extend(review.propose(S, attempt, today))
        if save(f"Exam {eid}: {pct(attempt['total'])}"):
            ss.exam = None
            ss.last_attempt = attempt
            if unanswered:
                st.toast(f"{unanswered} question(s) were left blank and counted as wrong.")
            st.rerun()


def page_exercises():
    st.title("Exercises")
    st.caption("Practice, not exam: these are cached and reused on purpose, unlike exam questions. "
               "Self-check against the hint and solution sketch — nothing here is graded.")
    ids = planlib.exam_ids(P)
    cur = planlib.current_entry(P)
    if ss.get("ex_week") not in ids:
        ss.ex_week = cur["id"]
    eid = st.selectbox("Week", ids, key="ex_week",
                       format_func=lambda i: f"{i} — {planlib.find(P, i)['title']}")
    entry = planlib.find(P, eid)
    pool = exlib.cached(S, eid)

    c1, c2 = st.columns(2)
    if c1.button("Generate 3 more (about $0.001)", type="primary" if not pool else "secondary"):
        try:
            with st.spinner("Writing exercises…"):
                new = exlib.generate(settings, S, entry, 3, entry.get("focus"), today=today)
            if new:
                exlib.add_to_cache(S, eid, new)
                save(f"Exercises for {eid}: +{len(new)} (cached, cost logged)")
            else:
                st.warning("The model didn't return usable exercises; try again.")
            st.rerun()
        except (llm.BudgetExceeded, llm.LLMError) as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"The OpenAI request failed: {e}")
    if pool and c2.button("Clear this week's cache"):
        S["exercises"][eid] = []
        save(f"Clear exercises {eid}")
        st.rerun()

    if not pool:
        st.write("No cached exercises for this week yet — generate some above.")
        return

    done_n = sum(1 for e in pool if e["done"])
    st.caption(f"{done_n} of {len(pool)} marked done for {eid}.")
    for e in pool:
        label = f"{'✅ ' if e['done'] else ''}[{e['kind']}] {e['prompt'][:80]}{'…' if len(e['prompt']) > 80 else ''}"
        with st.expander(label):
            st.markdown(e["prompt"])
            st.caption(f"Difficulty {e['difficulty']}/3 · added {e['created']}")
            with st.expander("Hint"):
                st.write(e["hint"] or "—")
            with st.expander("Solution sketch"):
                st.write(e["solution_sketch"] or "—")
            if st.checkbox("Done", value=e["done"], key=f"exdone_{e['id']}") != e["done"]:
                exlib.mark_done(S, eid, e["id"], not e["done"])
                save(f"Exercise done: {e['id']}")
                st.rerun()


def page_review_plan():
    st.title("Review & plan")
    pending = [p for p in S["proposals"] if p["status"] == "pending"]
    st.subheader("Proposed changes")
    if not pending:
        st.write("Nothing waiting. Take this week's exam to get proposals.")
    for p in pending:
        with st.container(border=True):
            who = "progress monitor" if p.get("by") == "monitor" else "exam"
            st.markdown(f"**{p['title']}** · from the {who} ({p['source']}) on {p['created']}")
            txt = st.text_area("Details (edit before accepting if you like)", p["detail"], key=f"pd_{p['id']}")
            a, r, _ = st.columns([1, 1, 4])
            if a.button("Accept", key=f"acc_{p['id']}", type="primary"):
                try:
                    review.apply(S, p, today, txt if txt != p["detail"] else None)
                    save(f"Accept: {p['title']}")
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
            if r.button("Reject", key=f"rej_{p['id']}"):
                review.reject(S, p, today)
                save(f"Reject: {p['title']}")
                st.rerun()

    accepted = [p for p in S["proposals"] if p["status"] == "accepted"][::-1][:10]
    if accepted:
        with st.expander("Accepted changes (undo)"):
            for p in accepted:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"{p.get('decided', '')} · **{p['title']}**")
                if c2.button("Undo", key=f"undo_{p['id']}"):
                    try:
                        review.undo(S, p, today); save(f"Undo: {p['title']}"); st.rerun()
                    except ValueError as err:
                        st.error(str(err))

    st.subheader("Jump ahead (or start somewhere else)")
    st.caption("Pick a later week and pass a short placement check to skip straight there — "
               f"{examlib.PLACEMENT_Q_PER_WEEK} fresh questions per week you'd skip, at most {examlib.PLACEMENT_MAX_WEEKS} weeks at once. "
               f"Score {pct(examlib.PLACEMENT_PASS)} or better and the skipped weeks are marked skipped; score lower and you're told which of "
               "them to actually start from.")
    targets = planlib.skippable_targets(P)
    placement = ss.get("placement")
    if placement and placement["target"] not in targets:
        placement = None
        ss.placement = None
    if not placement:
        if not targets:
            st.write("No later weeks to jump to.")
        else:
            tgt = st.selectbox("Jump to", targets, key="placement_target",
                               format_func=lambda i: f"{i} — {planlib.find(P, i)['title']}")
            if st.button("Take placement check"):
                try:
                    with st.spinner("Writing placement questions…"):
                        ss.placement = examlib.build_placement_exam(settings, S, tgt, today)
                    save(f"Placement check generated for {tgt} (cost logged)")
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
                except (llm.BudgetExceeded, llm.LLMError) as err:
                    st.error(str(err))
    else:
        result = ss.get("placement_result")
        if result:
            if result["passed"]:
                st.success(f"Scored {pct(result['total'])} — you can skip to {result['target']}.")
                if st.button("Confirm jump", type="primary"):
                    planlib.skip_to(S, result["target"], result["skipped"], today)
                    save(f"Skip to {result['target']}")
                    ss.placement = None
                    ss.placement_result = None
                    st.rerun()
            else:
                worst = ", ".join(result["weakest"])
                st.warning(f"Scored {pct(result['total'])} — below the {pct(examlib.PLACEMENT_PASS)} bar. "
                           f"Weakest on: {worst}. Consider starting there instead of {result['target']}.")
            if st.button("Try a different week"):
                ss.placement = None
                ss.placement_result = None
                st.rerun()
        else:
            with st.form("placement_form"):
                st.caption(f"Placement check for {placement['target']} — covering {', '.join(placement['skipped'])}.")
                p_answers = {}
                for n, q in enumerate(placement["mcq"], 1):
                    p_answers[q["id"]] = st.radio(f"**{n}.** {q['question']}", range(4), index=None,
                                                  format_func=lambda i, q=q: q["options"][i], key=f"p_{placement['id']}_{q['id']}")
                if st.form_submit_button("Submit placement check", type="primary"):
                    ss.placement_result = examlib.grade_placement(placement, p_answers, today)
                    st.rerun()
            if st.button("Cancel"):
                ss.placement = None
                st.rerun()

    st.subheader("Edit the plan yourself")
    ids = [e["id"] for e in P["entries"]]
    if ss.get("edit_sel") not in ids:
        ss.edit_sel = planlib.current_entry(P)["id"]
    eid = st.selectbox("Week to edit", ids, key="edit_sel")
    e = planlib.find(P, eid)
    with st.form(f"edit_{eid}"):
        title = st.text_input("Title", e["title"])
        math = st.text_area("Math track (one item per line)", "\n".join(e["math"]))
        build = st.text_area("Build track (one item per line)", "\n".join(e["build"]))
        deliv = st.text_input("Done when", e.get("deliverable", ""))
        notes = st.text_area("Adjustments / notes (one per line)", "\n".join(e.get("notes", [])))
        days = st.number_input("Length in days", 1, 21, int(e.get("days", 7)))
        if st.form_submit_button("Save changes"):
            lines = lambda s: [x.strip() for x in s.splitlines() if x.strip()]
            planlib.edit_entry(S, eid, {"title": title, "math": lines(math), "build": lines(build),
                                        "deliverable": deliv, "notes": lines(notes), "days": int(days)}, today)
            save(f"Edit {eid}")
            st.rerun()
    c1, c2, c3, c4 = st.columns(4)
    try:
        if c1.button("Move earlier"):
            planlib.move_entry(S, eid, -1, today); save(f"Move {eid}"); st.rerun()
        if c2.button("Move later"):
            planlib.move_entry(S, eid, +1, today); save(f"Move {eid}"); st.rerun()
        if c3.button("Insert extra week after"):
            new_end = planlib.end_date(P) + dt.timedelta(days=7)
            if new_end > planlib.max_end(P):
                st.error(f"That would end the plan on {new_end:%d %b %Y}, after the {planlib.max_end(P):%d %b %Y} limit. Shorten something first.")
            else:
                planlib.insert_extra_week(S, eid, e.get("focus", []), today); save(f"Insert after {eid}"); st.rerun()
        if c4.button("Remove week", disabled=e["kind"] not in ("extra", "rest")):
            planlib.remove_entry(S, eid, today); save(f"Remove {eid}"); st.rerun()
    except ValueError as err:
        st.error(str(err))

    st.subheader("History")
    if S["plan_history"]:
        st.dataframe(pd.DataFrame(S["plan_history"][::-1]), hide_index=True, width="stretch")


def page_queue():
    st.title("Review queue")
    st.caption("Reproduce each item from a blank page — derive it or code it — before marking it. "
               "Intervals: +1, +3, +7, +21, +60 days. A miss sends it back to +1 day.")
    with st.expander("Relearning protocol"):
        st.markdown("1. Re-read the source section.\n2. Re-derive with notes open.\n3. Re-implement from a blank file.\n"
                    "4. Tomorrow: closed-book attempt.\n5. The item restarts at +1 day.")
    due = spaced.due_items(S, today)
    if not due:
        st.success("Nothing due today.")
    for it in due:
        with st.container(border=True):
            tag = " · relearning" if it["relearn"] else ""
            try:
                topic = f"{it['entry']} — {planlib.find(P, it['entry'])['title']}"
            except KeyError:
                topic = it["entry"]
            st.markdown(f"**{it['text']}**  \n{topic} · step {it['step'] + 1} of {len(spaced.INTERVALS)}{tag} · due {it['due']}")
            a, b, _ = st.columns([1, 1, 3])
            if a.button("Reproduced it", key=f"ok_{it['id']}", type="primary"):
                spaced.mark_item(S, it["id"], True, today); save("Review: success"); st.rerun()
            if b.button("Couldn't — relearn", key=f"no_{it['id']}"):
                spaced.mark_item(S, it["id"], False, today); save("Review: relearn"); st.rerun()
    upcoming = sorted((i for i in S["review_items"].values() if i["due"] > today.isoformat()), key=lambda i: i["due"])[:20]
    if upcoming:
        st.subheader("Coming up")
        def _wk_title(eid):
            try:
                return f"{eid} — {planlib.find(P, eid)['title']}"
            except KeyError:
                return eid
        st.dataframe(pd.DataFrame([{"due": i["due"], "item": i["text"], "week": _wk_title(i["entry"])} for i in upcoming]),
                     hide_index=True, width="stretch")
    with st.form("add_item"):
        st.markdown("**Add your own item** (a derivation or implementation to keep fresh)")
        text = st.text_input("What to reproduce")
        if st.form_submit_button("Add") and text.strip():
            rid = f"own:{len(S['review_items'])}"
            S["review_items"][rid] = {"id": rid, "entry": planlib.current_entry(P)["id"], "text": text.strip(), "step": 0,
                                      "due": (today + dt.timedelta(days=1)).isoformat(), "relearn": False, "history": []}
            save("Add review item"); st.rerun()


def page_settings():
    st.title("Settings & cost")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", settings.openai_model)
    c2.metric("Spent this month", f"${llm.spent_this_month(S, today):.4f}")
    c3.metric("Monthly cap", f"${settings.monthly_budget_usd:.2f}")
    st.caption(f"Prices used for the cap: ${settings.price_input_per_m}/M input tokens, ${settings.price_output_per_m}/M output tokens. "
               "Check them against OpenAI's pricing page and update PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M in secrets.")
    st.caption(f"Progress is saved to: {store().kind}" + (f" ({settings.github_repo}/{settings.github_state_path})" if settings.use_github else ""))
    if S["cost_log"]:
        st.dataframe(pd.DataFrame(S["cost_log"][::-1][:100]), hide_index=True, width="stretch")
    st.download_button("Download all progress (JSON)", json.dumps(S, indent=1), "dl-coach-state.json", "application/json")
    if st.button("Reload from storage"):
        ss.pop("state", None); st.rerun()
    if st.button("Save again now"):
        save("Manual save")
    with st.expander("Latest allowed end date"):
        st.write("The coach never proposes a change that ends the plan after this date; it asks you to cut scope instead.")
        new_max = st.date_input("Latest end", planlib.max_end(P), key="max_end_in")
        if new_max != planlib.max_end(P) and st.button("Save latest end"):
            P["max_end_date"] = new_max.isoformat()
            planlib._log(S, "edit", f"Latest end set to {new_max:%d %b %Y}", today)
            save("Latest end date"); st.rerun()
    with st.expander("Start date"):
        st.write("Changing this shifts every week's dates by the same amount; each week keeps its own length.")
        cur_start = planlib._d(P["start_date"])
        new_start = st.date_input("Start date", cur_start, key="start_date_in")
        if new_start != cur_start and st.button("Save start date"):
            shifted_end = planlib.end_date(P) + (new_start - cur_start)
            if shifted_end > planlib.max_end(P):
                st.error(f"That would end the plan on {shifted_end:%d %b %Y}, after the {planlib.max_end(P):%d %b %Y} limit.")
            else:
                P["start_date"] = new_start.isoformat()
                planlib._log(S, "edit", f"Start date set to {new_start:%d %b %Y}", today)
                save("Start date"); st.rerun()
    with st.expander("Daily hour targets"):
        st.write("What the progress monitor treats as \"on pace\" each day — used for the planned-hours checks on the Dashboard.")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cur_hours = S.get("daily_hours") or list(monitor.DEFAULT_DAILY_HOURS)
        cols = st.columns(7)
        new_hours = [c.number_input(d, 0.0, 16.0, float(cur_hours[i]), 0.5, key=f"dh_{i}") for i, (c, d) in enumerate(zip(cols, days))]
        if new_hours != cur_hours and st.button("Save daily targets"):
            S["daily_hours"] = new_hours
            planlib._log(S, "edit", f"Daily hour targets set to {new_hours}", today)
            save("Daily hour targets"); st.rerun()
    with st.expander("Reset the plan"):
        st.write("Replaces the working plan with data/plan.yaml. Scores, questions and the review queue are kept.")
        confirm = st.text_input("Type RESET to confirm", key="reset_confirm")
        if st.button("Reset plan", disabled=confirm != "RESET"):
            S["plan"] = planlib.load_base_plan(BASE_PLAN)
            planlib._log(S, "reset", "Plan reset to data/plan.yaml", today)
            save("Reset plan"); st.rerun()


{"Dashboard": page_dashboard, "Weekly exam": page_exam, "Exercises": page_exercises, "Review & plan": page_review_plan,
 "Review queue": page_queue, "Settings & cost": page_settings}[page]()
