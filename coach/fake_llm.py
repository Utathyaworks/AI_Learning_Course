"""A stand-in for the OpenAI client, used by tests and by COACH_FAKE_LLM=1 for offline demos.

It returns well-formed questions without any network call or cost.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


class FakeClient:
    def __init__(self, n_questions: int = 12):
        self.n = n_questions
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        user = kwargs["messages"][-1]["content"]
        if '"strengths"' in user:
            body = {"strengths": ["Clean derivations"], "gaps": ["Bias correction in Adam"], "next_week_focus": ["Re-derive E[m_t]"]}
        else:
            qs = []
            for i in range(self.n):
                qs.append({
                    "question": f"Sample question {i + 1}: which statement is correct?",
                    "options": [f"Correct answer {i}", f"Wrong A {i}", f"Wrong B {i}", f"Wrong C {i}"],
                    "answer": 0,
                    "explanation": "The first option follows from the definition.",
                    "topic": ["linear maps", "broadcasting", "QR"][i % 3],
                    "concept": f"Idea number {i + 1}",
                    "difficulty": 1 + i % 3,
                })
            body = {"questions": qs}
            if "REVIEW questions" in user:
                self.last_review_prompt = user
                body["review"] = [dict(q, question="Review " + q["question"], topic="review") for q in qs[:8]]
        msg = SimpleNamespace(content=json.dumps(body))
        usage = SimpleNamespace(prompt_tokens=900, completion_tokens=2500)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
