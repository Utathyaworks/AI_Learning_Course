# DL Coach

A private study coach for Utathya's 61-week, depth-first deep-learning plan (5 Oct 2026 – 5 Dec 2027): Week 0, 35 topic weeks (including extra weeks on functions and their graphs, optimization formulation, geometry, probability, statistics, GPU computing, image/audio processing, vision tasks, attention mechanisms, modern architectures and mechanistic interpretability), 19 depth weeks (proofs + A100 reproduction + inside-the-model analysis), 3 consolidation weeks and 2 rest weeks, with a blog post every week. Every week it:

1. **Examines you** — a fresh exam every time: 12 multiple-choice questions on the week and up to 8 review
   questions that re-test the concepts you missed in your last exam plus your due revision items (both written by
   a small OpenAI model), and 2 from-scratch coding tasks scored by hidden tests. Questions are **never stored**:
   they live only in the browser tab until you submit; saved progress keeps scores per topic and a one-line note of
   each missed concept, which is what the next exam's review part is built from.
2. **Scores you** by part and by topic, shows explanations and reference solutions, and puts weak topics on a
   relearning schedule.
3. **Proposes plan changes** — advance, add relearning blocks, turn days into relearning, or insert an extra
   consolidation week — which you accept, edit or reject. You can also edit any week yourself.
4. **Keeps spaced revision** of each week's key results at +1, +3, +7, +21 and +60 days.

| Score | Proposal |
| --- | --- |
| ≥ 80% | Advance as planned |
| 65–79% | Add relearning blocks to next week |
| 50–64% | First two days of next week become relearning; topics flagged for the next consolidation week |
| < 50%, or two weeks under 65% | Insert an extra consolidation week (never past 16 Jan 2028 — then it asks you to cut scope) |

5. **Monitors progress — and changes nothing without your permission.** You log hours (with a one-line note),
   tick the week's checklist and paste the week's blog link on the Dashboard. Once a day the coach checks for:

   | Signal | Proposal |
   | --- | --- |
   | Exam not taken 2 days after the week ended | Extend that week; later weeks shift |
   | ≤ 2 days left and < 60% of the checklist, or < 50% of planned hours (your own daily targets, default 6 h Mon–Sat, set in Settings) | Extend by 3 days **or** keep dates and move unfinished items to the next depth week |
   | Exam ≥ 85% and checklist ≥ 90% with 2+ days left | Start the next week early |
   | > 90 h logged in 14 days, or three falling exam scores | Insert 3 rest days |

   Proposals wait on **Review & plan**: accept (optionally after editing the text), reject or ignore. Every accepted
   change snapshots the plan first, so the last 10 can be **undone**. A change that would end the plan after the
   latest end date (Settings) is refused and you're asked to cut scope.

Depth weeks (D1–D19) have their own exam: the review part focuses on weak topics from the weeks they cover, and the
coding part re-tests the two weakest tasks from those weeks. Rest weeks (R1, R2) have no exam.

6. **Lets you jump ahead — if you can prove it.** On **Review & plan**, pick any later week and take a placement
   check: 2 fresh questions per week you'd be skipping (at most 12 weeks at once). Score ≥ 70% and every skipped
   week is marked skipped and you land on the target week; score lower and the coach names the weeks you did worst
   on, so you know where to actually start instead. This is also how to start the plan "in the middle" if some of
   the early material is already familiar.

The plan's **start date** and your **daily hour targets** (Mon–Sun, default 6 h on weekdays/Saturday and 0 on
Sunday) are both editable on **Settings & cost** — changing the start date shifts every week's dates, keeping each
week's length; changing daily targets changes what the progress monitor (point 5 above) treats as "on pace."

## Cost

Only two things call the API: writing each exam (one call, ~1k input + ~4k output tokens) and the optional
written feedback. With a nano/mini-class model that is a fraction of a cent per week. The app prices every call
from the token counts OpenAI returns and **refuses any call that could push the month over `MONTHLY_BUDGET_USD`**
(default $2). Coding tasks cost nothing. Retaking an exam generates (and costs) a new one.

Model prices change: check OpenAI's pricing page and set `OPENAI_MODEL`, `PRICE_INPUT_PER_M` and
`PRICE_OUTPUT_PER_M` in secrets. The defaults assume `gpt-5-nano` at $0.05 / $0.40 per million tokens.

## Run it on your computer

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit: set OPENAI_API_KEY, delete the [auth] block and GITHUB_* lines, add DEV_MODE_NO_AUTH = true
streamlit run app.py
```

Without `GITHUB_*` settings, progress is saved to `data/state.json`. To try it without an API key or cost, run
`COACH_FAKE_LLM=1 streamlit run app.py` (questions are placeholders).

## Deploy (GitHub + Streamlit Community Cloud, your email only)

You need: a GitHub account, a Google account (the approved email), and a free Streamlit Community Cloud account
signed in with GitHub.

1. **Code repo.** Create a private repo, e.g. `dl-coach`, and push this folder to it.
2. **Data repo.** Create a second private repo, e.g. `dl-coach-data`, with a README so it has a `main` branch.
   Progress is saved there as commits. It is separate because Streamlit Cloud restarts the app on every push to
   the code repo.
3. **GitHub token.** GitHub → Settings → Developer settings → Fine-grained tokens → generate one with access to
   **only** `dl-coach-data`, permission *Contents: Read and write*. Copy it.
4. **Google sign-in.** Google Cloud Console → new project → *APIs & Services* → *OAuth consent screen*: External,
   publishing status **Testing**, add `utathya.course@gmail.com` as the only test user (a second lock: nobody else
   can sign in). Then *Credentials* → *Create credentials* → *OAuth client ID* → Web application, authorised
   redirect URI `https://YOUR-APP.streamlit.app/oauth2callback`. Copy the client ID and secret.
5. **Deploy.** share.streamlit.io → *Create app* → your `dl-coach` repo, branch `main`, file `app.py`. Choose the
   app URL (it must match the redirect URI). In *Advanced settings → Secrets*, paste the contents of
   `.streamlit/secrets.toml.example` with your real values. **Do not** include `DEV_MODE_NO_AUTH`.
6. **Make the app private.** In the app's *Share* settings, set it so only invited viewers can open it and invite
   only your email. (Community Cloud has limits on private apps per account; if yours is used up, the other two
   locks below still keep everyone else out.)

Three locks now protect it: Streamlit's private sharing, the Google consent screen's test-user list, and the
app's own `ALLOWED_EMAILS` check. If any secret is missing, the app refuses to open rather than running open.

## Using it each week

- **Sunday:** publish the blog post and paste its link on the Dashboard; then *Weekly exam* → Start exam → answer closed-book → Submit. Read the explanations and compare your
  code with the reference solution.
- **Review & plan:** accept, edit or reject the proposals; edit any week if your situation changes.
- **Every day:** *Dashboard* → log hours and a note, tick finished checklist items (10 seconds).
- **Every day:** *Review queue* → reproduce each due item from a blank page, then mark it.
- **Consolidation weeks (C1, C2, C3):** the exam focuses on your weakest topics so far, with C1 adding a Slurm task.

## Changing the content

- **Plan:** `data/plan.yaml` is the starting plan. The app keeps its own working copy (with your accepted changes)
  in the data repo; *Settings → Reset the plan* reloads the YAML.
- **Coding tasks:** they are defined in `dev/tasks_part*.py`. After editing, run `python dev/build_tasks.py`; it
  rewrites `tasks/` and checks that each reference solution passes and each starter fails.
- **Tests:** `python -m pytest -q tests` (also runs in GitHub Actions on every push).

## Security notes

- Your code submissions run on the app's server in a separate process with a 60-second limit and without the
  app's secrets in its environment. Only you can reach the app, so only your own code runs there.
- Reference solutions live in `tasks/*/solution.py` in your private repo. The app shows them only after you submit.
