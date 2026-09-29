"""Comparison website.  Run locally with:  streamlit run app.py"""
import json
import re
import time
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

from policybot import config
from policybot.engine import METHODS, answer

ROOT = Path(__file__).parent
st.set_page_config(page_title="Policy Assistant Comparison", layout="wide")

st.title("Company Policy Assistant")
st.caption(f"Rules-based search vs. LLM without vector index vs. LLM with vector index · "
           f"{config.STUDENT_NAME} · model: {config.GEMINI_MODEL}")

BADGE = {"grounded": ("Supported by policy", "green"), "abstained": ("Declined (not covered)", "blue"),
         "unsupported": ("Unsupported claim", "red"), "error": ("Error", "gray")}

EXAMPLES = [
    "How many vacation days do I get per year?",
    "Can I work from my family's place in Spain for a month?",
    "How many sick days do I accrue each month?",
    "Who pays for my certification course?",
    "Can I bring my dog to the office?",
]


# ---- Protect the API quota on a public site ----
SESSION_LIMIT = int(config.get_setting("SESSION_LIMIT", "15"))   # questions per visitor
HOURLY_LIMIT = int(config.get_setting("HOURLY_LIMIT", "60"))     # questions per hour, all visitors


@st.cache_resource
def _global_log() -> deque:
    return deque()


def limit_message() -> str | None:
    now, log = time.time(), _global_log()
    while log and now - log[0] > 3600:
        log.popleft()
    if st.session_state.get("asked", 0) >= SESSION_LIMIT:
        return f"You have reached the limit of {SESSION_LIMIT} questions for this visit."
    if len(log) >= HOURLY_LIMIT:
        return "The site has reached its hourly question limit. Please try again later."
    st.session_state["asked"] = st.session_state.get("asked", 0) + 1
    log.append(now)
    return None


class _Uncached(Exception):
    def __init__(self, result: dict):
        self.result = result


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_answer(question: str, method: str) -> dict:
    r = answer(question, method).to_dict()
    if r["error"]:
        raise _Uncached(r)  # don't cache errors such as rate limits
    return r


def cached_answer(question: str, method: str) -> dict:
    try:
        return _cached_answer(question, method)
    except _Uncached as e:
        return e.result


tab_ask, tab_compare = st.tabs(["Ask a question", "Comparison and conclusion"])

with tab_ask:
    example = st.selectbox("Example questions", ["(write your own)"] + EXAMPLES)
    question = st.text_input("Policy question", value="" if example.startswith("(") else example,
                             max_chars=300)
    clicked = st.button("Ask all three", type="primary") and question.strip()
    blocked = limit_message() if clicked else None
    if blocked:
        st.warning(blocked)
    elif clicked:
        cols = st.columns(3)
        for col, (method, (label, _)) in zip(cols, METHODS.items()):
            with col:
                st.subheader(label)
                with st.spinner("Thinking..."):
                    r = cached_answer(question.strip(), method)
                text, colour = BADGE.get(r["grounding"], ("?", "gray"))
                st.markdown(f":{colour}-background[{text}]")
                st.write(r["answer"])
                st.markdown(f"**Relevant policy:** {r['policy_title'] or '—'}")
                if r["policy_text"]:
                    st.caption(f"Policy text: “{r['policy_text']}”")
                c1, c2 = st.columns(2)
                c1.metric("Response time", f"{r['latency_ms']:.0f} ms")
                c2.metric("Tokens", f"{r['total_tokens']:,}")
                with st.expander("Details"):
                    st.write(f"Check: {r['grounding_reason']}")
                    if r["input_tokens"]:
                        st.write(f"Input {r['input_tokens']:,} · output {r['output_tokens']:,}"
                                 + (f" · query embedding ≈{r['embedding_tokens_est']}" if r["embedding_tokens_est"] else ""))
                    if r["retrieved"]:
                        st.write("Candidates considered: " + "; ".join(r["retrieved"]))
        st.caption("The live badge is an automatic check (cited policy exists, no numbers that are not in "
                   "the policy). The evaluation tab uses a stricter LLM judge over a fixed test set.")

with tab_compare:
    summary_file = ROOT / "results" / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        st.markdown("**Table 1. Results on the 28-question test set**")
        st.dataframe(pd.DataFrame(summary["overall"]), hide_index=True, use_container_width=True)
        st.markdown("**Table 2. Correct policy and supported answer, by question type (%)**")
        st.dataframe(pd.DataFrame(summary["by_type"]), hide_index=True, use_container_width=True)
    else:
        st.info("Run `python evaluate.py` to generate the results tables.")

    comparison = ROOT / "COMPARISON.md"
    if comparison.exists():
        text = re.sub(r"<!--.*?-->", "", comparison.read_text(encoding="utf-8"), flags=re.S)
        st.markdown(text)

    details = ROOT / "results" / "eval_results.csv"
    if details.exists():
        st.download_button("Download per-question results (CSV)", details.read_bytes(),
                           "eval_results.csv", "text/csv")
