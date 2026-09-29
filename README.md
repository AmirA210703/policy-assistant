# Company Policy Assistant

A bot that answers company policy questions and names the relevant policy, built three ways and compared:

| Approach | How it works | File |
|---|---|---|
| Rules-based search | BM25 keyword ranking + synonym rules, returns the policy text verbatim | `policybot/rules.py` |
| LLM without vector index | Gemini gets all 98 policies in the prompt | `policybot/llm.py` → `answer_full_context` |
| LLM with vector index | Gemini embeddings, top-4 cosine similarity, only those policies in the prompt | `policybot/llm.py` → `answer_rag` |

The website (`app.py`, Streamlit) shows each approach's answer, relevant policy, response time, token use and whether the answer is supported by the policy database. The Slack bot (`slack_bot.py`) uses the same engine.

- Live site: https://policy-assistant-d512.onrender.com
- Models (Gemini API, free tier): answers `gemini-3.5-flash-lite`, evaluation judge `gemini-3.1-flash-lite`, embeddings `gemini-embedding-001`.
  The judge uses a different model so it has its own daily quota and does not grade its own answers.

## Project structure

```
data/company_policies.csv     policy database (98 policies)
data/test_questions.csv       28 test questions with expected policy (NONE = not covered)
data/policy_embeddings.npz    cached vector index (created on first run - commit it)
policybot/                    shared engine used by the website, the bot and the evaluation
evaluate.py                   runs the test set, writes results/
app.py                        comparison website
slack_bot.py                  Slack bot (Socket Mode, runs locally)
COMPARISON.md                 the two-paragraph comparison shown on the website
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your Gemini API key into .env
```

Get a key at https://aistudio.google.com/apikey. If a model name has been retired, change
`GEMINI_MODEL` / `GEMINI_JUDGE_MODEL` / `GEMINI_EMBEDDING_MODEL` in `.env` (see https://ai.google.dev/gemini-api/docs/models).
Free-tier daily limits differ a lot per model (see https://aistudio.google.com/rate-limit). The Flash Lite models allow
about 500 requests per day, while the full Flash models allow 20, which is too few for the evaluation (about 112 calls).
Models that reject `thinking_budget` are detected automatically, and the code retries without it.

## 1. Run the evaluation

```bash
python evaluate.py --sleep 4           # --sleep helps with free-tier rate limits
python evaluate.py --sleep 4           # run again to resume after a stop (quota, 503 overload)
python evaluate.py --fresh --sleep 4   # start over from scratch
python evaluate.py --summary-only      # rebuild results/summary.json from the saved CSV
```

This writes `results/eval_results.csv` (every answer) and `results/summary.json` (the two tables on
the website), and creates `data/policy_embeddings.npz`. Progress is saved after every answer. If the daily quota
runs out or the API is overloaded, the run stops cleanly, and the next run continues where it left off.
Response time in Table 1 is the median, because retries during API overload create a few extreme values.

How "unsupported" is measured:
- **Live on the website:** an automatic check. The cited policy has to exist, and the answer must not contain numbers the policy does not state.
- **In the evaluation:** an LLM judge compares each answer with the cited policy text and flags any added fact.
- The rules-based search returns policy text verbatim, so it is supported by construction. Its errors show up as *wrong policy* instead.

Limitation: the synonym rules in `rules.py` were written with the test questions in mind, so the
rules-based score is optimistic. Adding a few questions written by someone else gives a fairer test.

## 2. Run and deploy the website

```bash
streamlit run app.py
```

### Deploy on Render (free)

`render.yaml` describes the service, so Render needs no manual build settings.

1. Run `python evaluate.py` first. Then push this repo to GitHub, including `results/` and `data/policy_embeddings.npz`, but not `.env`.
2. On https://dashboard.render.com, choose **New → Blueprint** and connect the GitHub repo. Render reads `render.yaml`.
3. When asked, fill in `GEMINI_API_KEY` and `STUDENT_NAME`, then click **Apply**.
4. After the build (a few minutes) the site is live at `https://policy-assistant-xxxx.onrender.com`. That is the link you hand in.

Each `git push` redeploys the site automatically.

Notes on the free plan:
- The service sleeps after about 15 minutes without visitors, and the first visit afterwards takes 30–60 seconds. Open the site yourself shortly before it is graded.
- The disk is not persistent, so commit `data/policy_embeddings.npz` and `results/`. Otherwise the vector index is rebuilt on every restart.
- Measured response times include Render's network. The numbers in the tables come from `evaluate.py` on your own machine, so say so in the text.

Alternative: Streamlit Community Cloud (https://share.streamlit.io). There you add the key under **Advanced settings → Secrets**.

## 3. Slack bot

1. Create a new Slack workspace whose name identifies you, and a channel such as `#policy-bot-test`.
2. Go to https://api.slack.com/apps and choose **Create New App → From a manifest**. Pick the workspace, then paste `slack_manifest.yaml`.
3. Under **Basic Information → App-Level Tokens**, create a token with the `connections:write` scope. It starts with `xapp-` and goes in `.env` as `SLACK_APP_TOKEN`.
4. Under **Install App**, install the app to the workspace, and copy the Bot User OAuth Token (`xoxb-`) into `.env` as `SLACK_BOT_TOKEN`.
5. Run `python slack_bot.py`, then run `/invite @PolicyBot` in the channel.
6. Ask a question, e.g. `@PolicyBot How many vacation days do I get?`
7. Invite raz@sdu.dk to the workspace and add them to the channel.

`BOT_METHOD` in `.env` selects the approach the bot uses. The default is `llm_rag`.
