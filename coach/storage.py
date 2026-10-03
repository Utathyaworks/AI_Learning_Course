"""Where the coach keeps its state: a JSON file, either in the GitHub repo or on local disk.

Streamlit Community Cloud wipes local files on every restart, so the deployed
app saves state back to the repo through the GitHub contents API. Locally
(and in tests) a plain file is used.
"""
from __future__ import annotations

import base64
import copy
import json
import os
from typing import Any

import requests

EMPTY_STATE: dict[str, Any] = {
    "version": 1,
    "plan": None,  # working copy of the plan; created from data/plan.yaml on first run
    "plan_history": [],  # list of {date, change, detail}
    "attempts": [],  # one per submitted exam
    "last_exam": None,  # {entry, date, missed: ["topic: concept", ...], weak_topics} — drives the next review part
    "review_items": {},  # key-result id -> spaced-revision record
    "proposals": [],  # pending plan-change proposals
    "cost_log": [],  # one per API call
    "ai_feedback": {},  # attempt id -> feedback dict
    "daily_log": [],  # {date, entry, hours, note}
    "checklist": {},  # entry id -> list of ticked plan items
    "blog": {},  # entry id -> URL of that week's post
    "snapshots": [],  # plan copies taken before each accepted change (for undo)
    "daily_hours": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0],  # target hours Mon..Sun, used for pace checks
    "exercises": {},  # entry id -> cached list of practice exercises (unlike exam questions, these ARE kept)
}


def new_state() -> dict[str, Any]:
    return copy.deepcopy(EMPTY_STATE)


def _fill_defaults(state: dict[str, Any]) -> dict[str, Any]:
    for k, v in EMPTY_STATE.items():
        state.setdefault(k, copy.deepcopy(v))
    for legacy in ("question_bank", "question_stats"):  # questions are never stored
        state.pop(legacy, None)
    return state


class StorageError(RuntimeError):
    pass


class LocalStore:
    kind = "local file"

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return new_state()
        with open(self.path, encoding="utf-8") as f:
            return _fill_defaults(json.load(f))

    def save(self, state: dict[str, Any], message: str = "Update coach state") -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


class GitHubStore:
    """Reads and writes one JSON file in a GitHub repo (contents API)."""

    kind = "GitHub repo"
    api = "https://api.github.com"

    def __init__(self, token: str, repo: str, path: str, branch: str = "main", session=None):
        self.token, self.repo, self.path, self.branch = token, repo, path, branch
        self.http = session or requests.Session()
        self._sha: str | None = None

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _url(self):
        return f"{self.api}/repos/{self.repo}/contents/{self.path}"

    def _get(self):
        r = self.http.get(self._url(), headers=self._headers(), params={"ref": self.branch}, timeout=20)
        if r.status_code == 404:
            return None, None
        if r.status_code != 200:
            raise StorageError(f"GitHub read failed ({r.status_code}): {r.text[:200]}")
        body = r.json()
        content = base64.b64decode(body["content"]).decode("utf-8")
        return json.loads(content), body["sha"]

    def load(self) -> dict[str, Any]:
        data, sha = self._get()
        self._sha = sha
        return _fill_defaults(data) if data else new_state()

    def save(self, state: dict[str, Any], message: str = "Update coach state") -> None:
        payload = json.dumps(state, indent=1, ensure_ascii=False).encode("utf-8")
        for attempt in range(2):
            body = {
                "message": message,
                "content": base64.b64encode(payload).decode("ascii"),
                "branch": self.branch,
            }
            if self._sha:
                body["sha"] = self._sha
            r = self.http.put(self._url(), headers=self._headers(), json=body, timeout=30)
            if r.status_code in (200, 201):
                self._sha = r.json()["content"]["sha"]
                return
            if r.status_code in (409, 422) and attempt == 0:
                # Someone (or another tab) saved in between: refresh the sha and retry once.
                _, self._sha = self._get()
                continue
            raise StorageError(f"GitHub save failed ({r.status_code}): {r.text[:200]}")


def make_store(settings):
    if settings.use_github:
        return GitHubStore(settings.github_token, settings.github_repo, settings.github_state_path, settings.github_branch)
    return LocalStore(settings.local_state_path)
