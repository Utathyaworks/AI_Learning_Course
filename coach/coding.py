"""Coding tasks: a fixed bank under tasks/, scored by hidden pytest tests.

Each task folder holds:
  task.md      what to implement (shown during the exam)
  starter.py   the function signatures to fill in (shown during the exam)
  test_task.py hidden tests; they import from `submission`
  solution.py  reference solution, shown only after the exam is submitted
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


@dataclass
class Task:
    id: str
    title: str
    prompt: str
    starter: str
    tests: str
    solution: str


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_task(task_id: str, tasks_dir: str = TASKS_DIR) -> Task:
    d = os.path.join(tasks_dir, task_id)
    prompt = _read(os.path.join(d, "task.md"))
    title = prompt.splitlines()[0].lstrip("# ").strip()
    return Task(
        id=task_id,
        title=title,
        prompt=prompt,
        starter=_read(os.path.join(d, "starter.py")),
        tests=_read(os.path.join(d, "test_task.py")),
        solution=_read(os.path.join(d, "solution.py")),
    )


def all_task_ids(tasks_dir: str = TASKS_DIR) -> list[str]:
    return sorted(
        t for t in os.listdir(tasks_dir) if os.path.isfile(os.path.join(tasks_dir, t, "task.md"))
    )


@dataclass
class RunResult:
    passed: int
    total: int
    failures: list[str]
    output: str
    timed_out: bool = False

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_tests(code: str, tests: str, timeout: int = 60) -> RunResult:
    """Run hidden tests against `code` in a separate Python process with a time limit."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "submission.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(tmp, "test_task.py"), "w", encoding="utf-8") as f:
            f.write(tests)
        report = os.path.join(tmp, "report.xml")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}", "test_task.py"]
        # Minimal environment: submitted code must not see the app's secrets (API key, GitHub token).
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "VIRTUAL_ENV")
        env = {k: os.environ[k] for k in keep if k in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, env=env)
            out = (proc.stdout + proc.stderr)[-6000:]
        except subprocess.TimeoutExpired:
            return RunResult(0, _count_tests(tests), ["Time limit exceeded"], "", timed_out=True)
        if not os.path.exists(report):
            return RunResult(0, _count_tests(tests), ["Tests could not run (syntax or import error)"], out)
        return _parse_junit(report, out, tests)


def _count_tests(tests: str) -> int:
    return max(1, sum(1 for line in tests.splitlines() if line.startswith("def test_")))


def _parse_junit(path: str, out: str, tests: str) -> RunResult:
    root = ET.parse(path).getroot()
    cases = root.iter("testcase")
    total, passed, failures = 0, 0, []
    for c in cases:
        total += 1
        failed = any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))
        if failed:
            failures.append(c.get("name", "?"))
        else:
            passed += 1
    if total == 0:  # collection error: the module failed to import
        return RunResult(0, _count_tests(tests), ["Tests could not run (syntax or import error)"], out)
    return RunResult(passed, total, failures, out)
