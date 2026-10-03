"""Write the coding-task bank into tasks/ and verify every task.

For each task: the reference solution must pass every hidden test, and the
starter must fail at least one (so an untouched starter scores below 100%).

Run from the repo root:  python dev/build_tasks.py
"""
import importlib.util
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from coach.coding import run_tests  # noqa: E402

HEADER = "import numpy as np\nfrom submission import *  # noqa: F401,F403\n"


def load_parts():
    tasks = []
    for name in sorted(os.listdir(os.path.join(ROOT, "dev"))):
        if name.startswith("tasks_part") and name.endswith(".py"):
            spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(ROOT, "dev", name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tasks.extend(mod.T)
    return tasks


def main(only=None):
    tasks = load_parts()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    out_dir = os.path.join(ROOT, "tasks")
    bad = []
    for t in tasks:
        if only and t["id"] not in only:
            continue
        d = os.path.join(out_dir, t["id"])
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        tests = HEADER + t["tests"].lstrip("\n")
        files = {
            "task.md": f"# {t['title']}\n\n{t['prompt'].strip()}\n",
            "starter.py": t["starter"],
            "solution.py": t["solution"],
            "test_task.py": tests,
        }
        for fn, content in files.items():
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                f.write(content)
        good = run_tests(t["solution"], tests)
        start = run_tests(t["starter"], tests)
        ok = good.passed == good.total and good.total > 0 and start.score < 1.0
        print(f"{'OK  ' if ok else 'FAIL'} {t['id']:32s} solution {good.passed}/{good.total}  starter {start.passed}/{start.total}")
        if not ok:
            bad.append(t["id"])
            print(good.output[-2500:])
    print(f"\n{len(tasks)} tasks, {len(bad)} failing: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:]) or None))
