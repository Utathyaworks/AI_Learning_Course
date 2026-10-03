import datetime as dt
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coach import coding, llm, review, spaced, storage  # noqa: E402
from coach import exam as examlib  # noqa: E402
from coach import plan as planlib  # noqa: E402
from coach.config import Settings  # noqa: E402
from coach.fake_llm import FakeClient  # noqa: E402

BASE = os.path.join(ROOT, "data", "plan.yaml")
D0 = dt.date(2026, 10, 4)


def fresh_state():
    s = storage.new_state()
    planlib.ensure_plan(s, BASE)
    return s


def settings(**kw):
    return Settings(openai_api_key="test", **kw)


# ---- plan ----------------------------------------------------------------------

def test_plan_dates_match_the_doc():
    p = planlib.load_base_plan(BASE)
    sch = {e["id"]: e for e in planlib.schedule(p)}
    d = lambda y, m, dd: dt.date(y, m, dd)
    expect = {"W00": d(2026, 10, 1), "W01": d(2026, 10, 5), "D1": d(2026, 10, 19), "W02": d(2026, 10, 26),
              "W02B": d(2026, 11, 2), "R1": d(2026, 12, 28), "D4": d(2027, 1, 4), "W06A": d(2027, 1, 11), "W06": d(2027, 1, 18), "C1": d(2027, 3, 8),
              "W10": d(2027, 3, 15), "D11": d(2027, 6, 14), "W15A": d(2027, 6, 21), "W15B": d(2027, 6, 28), "D12": d(2027, 7, 5),
              "C2": d(2027, 7, 12), "R2": d(2027, 7, 19), "W16": d(2027, 7, 26), "W19B": d(2027, 9, 6),
              "D15": d(2027, 9, 13), "C3": d(2027, 10, 4), "D19": d(2027, 11, 15), "W24": d(2027, 11, 22)}
    for k, v in expect.items():
        assert sch[k]["start"] == v, k
    assert sch["W25"]["end"] == d(2027, 12, 5)
    assert planlib.max_end(p) == d(2028, 1, 16)
    assert len(p["entries"]) == 62
    kinds = [e["kind"] for e in p["entries"]]
    assert kinds.count("depth") == 19 and kinds.count("rest") == 2
    for e in p["entries"]:
        if e["kind"] == "depth":
            assert e["covers"] and all(c in sch for c in e["covers"]) and e["math"] and e["build"], e["id"]


def test_every_plan_task_exists_and_state_is_json():
    p = planlib.load_base_plan(BASE)
    ids = set(coding.all_task_ids())
    for e in p["entries"]:
        for t in e["tasks"]:
            assert t in ids, t
    json.dumps(fresh_state())  # dates must not leak into the state


def test_insert_week_shifts_later_dates():
    s = fresh_state()
    before = planlib.schedule(s["plan"])
    new = planlib.insert_extra_week(s, "W03", ["SGD"], D0)
    after = {e["id"]: e for e in planlib.schedule(s["plan"])}
    assert after[new]["start"] == dt.date(2026, 11, 16)
    assert after["D2"]["start"] == dt.date(2026, 11, 23)
    assert planlib.end_date(s["plan"]) == before[-1]["end"] + dt.timedelta(days=7)
    planlib.remove_entry(s, new, D0)
    assert planlib.end_date(s["plan"]) == before[-1]["end"]
    with pytest.raises(ValueError):
        planlib.remove_entry(s, "W04", D0)


# ---- spaced revision -----------------------------------------------------------

def test_spaced_intervals_and_relearn():
    s = fresh_state()
    e = planlib.find(s["plan"], "W01")
    spaced.add_key_results(s, e, D0)
    rid = "W01:0"
    assert s["review_items"][rid]["due"] == "2026-10-05"
    day = D0
    expected = [3, 7, 21, 60, 60]
    for gap in expected:
        day = dt.date.fromisoformat(s["review_items"][rid]["due"])
        spaced.mark_item(s, rid, True, day)
        assert dt.date.fromisoformat(s["review_items"][rid]["due"]) - day == dt.timedelta(days=gap)
    spaced.mark_item(s, rid, False, day)
    it = s["review_items"][rid]
    assert it["relearn"] and it["step"] == 0 and it["due"] == (day + dt.timedelta(days=1)).isoformat()


def test_review_concepts_come_from_last_exam_and_due_items():
    s = fresh_state()
    s["last_exam"] = {"entry": "W01", "missed": ["QR: positive diagonal of R"], "weak_topics": []}
    spaced.add_key_results(s, planlib.find(s["plan"], "W01"), D0)
    c = examlib.review_concepts(s, D0 + dt.timedelta(days=1))
    assert c[0] == "QR: positive diagonal of R" and len(c) == 1 + len(planlib.find(s["plan"], "W01")["key_results"])


# ---- budget --------------------------------------------------------------------

def test_budget_cap_blocks_calls():
    s = fresh_state()
    s["cost_log"].append({"date": "2026-10-02", "purpose": "x", "model": "m", "in_tokens": 0, "out_tokens": 0, "usd": 1.999})
    with pytest.raises(llm.BudgetExceeded):
        llm.generate_mcqs(settings(), s, planlib.find(s["plan"], "W01"), client=FakeClient(), today=dt.date(2026, 10, 3))
    # a new month starts fresh
    qs = llm.generate_mcqs(settings(), s, planlib.find(s["plan"], "W01"), client=FakeClient(), today=dt.date(2026, 11, 1))
    assert len(qs) == 12
    assert llm.spent_this_month(s, dt.date(2026, 11, 1)) == pytest.approx(900 / 1e6 * 0.05 + 2500 / 1e6 * 0.40)


def test_mcq_validation_shuffles_and_drops_bad():
    raw = {"questions": [
        {"question": "q", "options": ["a", "b", "c", "d"], "answer": 2, "topic": "t"},
        {"question": "bad", "options": ["a", "b"], "answer": 0},
        {"question": "dup", "options": ["a", "a", "c", "d"], "answer": 0},
    ] + [{"question": f"x{i}", "options": ["w", "x", "y", "z"], "answer": 1, "topic": "t"} for i in range(4)]}
    qs = llm.validate_mcqs(raw, "W01", seed=3)
    assert len(qs) == 5
    for q in qs:
        assert q["options"][q["answer"]] in {"c", "x"}


# ---- exam end to end -----------------------------------------------------------

def _solve(task_ids):
    return {t: coding.load_task(t).solution for t in task_ids}


def test_full_exam_cycle_with_perfect_answers():
    s = fresh_state()
    cfg = settings()
    fc = FakeClient()
    ex = examlib.build_exam(cfg, s, "W01", D0, client=fc)
    assert len(ex["mcq"]) == 12 and len(ex["tasks"]) == 2 and ex["spaced"] == []
    answers = {q["id"]: q["answer"] for q in ex["mcq"]}
    att = examlib.grade(ex, answers, _solve([t["id"] for t in ex["tasks"]]), D0)
    assert att["total"] == pytest.approx(1.0)
    examlib.record(s, att, D0)
    s["proposals"].extend(review.propose(s, att, D0))
    assert s["proposals"][-1]["type"] == review.ADVANCE
    assert planlib.find(s["plan"], "W01")["status"] == "done"
    # questions are never stored: no question text or options anywhere in the saved state
    blob = json.dumps(s)
    assert "Sample question" not in blob and "Correct answer" not in blob and "question_bank" not in blob
    # every exam is generated fresh
    examlib.build_exam(cfg, s, "W01", D0, client=fc)
    assert fc.calls == 2


def test_review_part_targets_last_exams_misses():
    s = fresh_state()
    fc = FakeClient()
    ex = examlib.build_exam(settings(), s, "W01", D0, client=fc)
    answers = {q["id"]: q["answer"] for q in ex["mcq"]}
    first = ex["mcq"][0]
    answers[first["id"]] = (first["answer"] + 1) % 4          # miss one question
    att = examlib.grade(ex, answers, _solve([t["id"] for t in ex["tasks"]]), D0)
    examlib.record(s, att, D0)
    assert s["last_exam"]["missed"] == [f"{first['topic']}: {first['concept']}"]
    ex2 = examlib.build_exam(settings(), s, "W02", D0, client=fc)   # same day: key results not yet due
    assert first["concept"] in fc.last_review_prompt
    assert len(ex2["spaced"]) == 2 and all(q["question"].startswith("Review") for q in ex2["spaced"])


def test_low_score_proposes_and_applies_extra_week():
    s = fresh_state()
    ex = examlib.build_exam(settings(), s, "W01", D0, client=FakeClient())
    wrong = {q["id"]: (q["answer"] + 1) % 4 for q in ex["mcq"]}
    att = examlib.grade(ex, wrong, {}, D0)  # starters fail every test
    assert att["total"] == 0.0 and att["weak_topics"]
    examlib.record(s, att, D0)
    props = review.propose(s, att, D0)
    assert props[0]["type"] == review.INSERT_WEEK
    old_end = planlib.end_date(s["plan"])
    review.apply(s, props[0], D0)
    assert planlib.end_date(s["plan"]) == old_end + dt.timedelta(days=7)
    assert planlib.index_of(s["plan"], "X1") == planlib.index_of(s["plan"], "W01") + 1
    assert any(i["relearn"] for i in s["review_items"].values())


def test_middle_scores_add_notes():
    s = fresh_state()
    att = {"id": "a1", "entry": "W02", "total": 0.72, "weak_topics": ["QR"]}
    p = review.propose(s, att, D0)[0]
    assert p["type"] == review.RELEARN_BLOCKS
    review.apply(s, p, D0)
    assert any("QR" in n for n in planlib.find(s["plan"], "W02B")["notes"])
    att2 = {"id": "a2", "entry": "W07", "total": 0.55, "weak_topics": ["init"]}
    s["attempts"] = [{"id": "a0", "total": 0.9}]
    p2 = review.propose(s, att2, D0)[0]
    assert p2["type"] == review.RELEARN_DAYS
    review.apply(s, p2, D0)
    assert "init" in planlib.find(s["plan"], "C1")["focus"]


def test_two_low_weeks_in_a_row_insert_week():
    s = fresh_state()
    s["attempts"] = [{"id": "a0", "total": 0.6}]
    p = review.propose(s, {"id": "a1", "entry": "W05", "total": 0.62, "weak_topics": []}, D0)[0]
    assert p["type"] == review.INSERT_WEEK


def test_cut_scope_when_past_max_end():
    s = fresh_state()
    s["plan"]["max_end_date"] = "2027-11-08"
    p = review.propose(s, {"id": "a1", "entry": "W05", "total": 0.3, "weak_topics": ["x"]}, D0)[0]
    assert p["type"] == review.CUT_SCOPE


def test_consolidation_exam_picks_weakest_tasks():
    s = fresh_state()
    s["attempts"] = [{"entry": "W01", "by_topic": {"QR": 0.2}, "coding": {"items": [
        {"task_id": "w01_gram_schmidt", "score": 1.0}, {"task_id": "w01_broadcast_shape", "score": 0.2}]}}]
    ex = examlib.build_exam(settings(), s, "C1", D0, client=FakeClient())
    ids = [t["id"] for t in ex["tasks"]]
    assert ids[0] == "c1_slurm_array" and len(ids) == 2 and "w01_gram_schmidt" not in ids


# ---- coding runner -------------------------------------------------------------

def test_runner_handles_syntax_errors_and_timeouts():
    t = coding.load_task("w01_broadcast_shape")
    r = coding.run_tests("def broadcast_shape(:\n", t.tests)
    assert r.score == 0.0
    r = coding.run_tests("def broadcast_shape(*s):\n    while True: pass\n", t.tests, timeout=5)
    assert r.timed_out and r.score == 0.0


# ---- storage -------------------------------------------------------------------

def test_local_store_roundtrip(tmp_path):
    st = storage.LocalStore(str(tmp_path / "d" / "state.json"))
    s = fresh_state()
    s["cost_log"].append({"date": "2026-10-01", "usd": 0.1})
    st.save(s)
    assert st.load()["cost_log"][0]["usd"] == 0.1


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


class FakeGitHub:
    def __init__(self):
        self.files, self.sha, self.puts = None, None, 0

    def get(self, url, headers=None, params=None, timeout=None):
        assert headers["Authorization"].startswith("Bearer ")
        if self.files is None:
            return _Resp(404, {})
        return _Resp(200, {"content": self.files, "sha": self.sha})

    def put(self, url, headers=None, json=None, timeout=None):
        self.puts += 1
        if self.sha and json.get("sha") != self.sha:
            return _Resp(409, {"message": "conflict"})
        self.files, self.sha = json["content"], f"sha{self.puts}"
        return _Resp(201, {"content": {"sha": self.sha}})


def test_github_store_roundtrip_and_conflict_retry():
    gh = FakeGitHub()
    st = storage.GitHubStore("tok", "me/repo", "data/state.json", session=gh)
    s = st.load()
    s["attempts"].append({"x": 1})
    st.save(s)
    st2 = storage.GitHubStore("tok", "me/repo", "data/state.json", session=gh)
    assert st2.load()["attempts"] == [{"x": 1}]
    st2.save(s)                 # moves the remote sha
    st.save(s)                  # stale sha -> 409 -> refresh -> success
    assert gh.puts == 4


def test_submissions_cannot_read_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    tests = "import os\n\ndef test_env():\n    assert 'OPENAI_API_KEY' not in os.environ and 'GITHUB_TOKEN' not in os.environ\n"
    assert coding.run_tests("", tests).score == 1.0
