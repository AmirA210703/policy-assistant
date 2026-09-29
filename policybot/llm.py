"""Approaches 2 and 3: Gemini without a vector index (whole policy database in the prompt)
and Gemini with a vector index (only the top-k retrieved policies in the prompt).

Both use the same instructions so the only difference is how the context is chosen.
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config, gemini, grounding
from .policies import ROOT, load_policies
from .result import Result

SYSTEM_PROMPT = """You are the company's internal policy assistant.
Answer ONLY from the policies provided below. Each policy is written as [Title] (Category) text.

Rules:
1. Pick the single most relevant policy and give its exact title in "policy_title".
2. Answer in 1-3 sentences using only what that policy says.
3. If the policy is relevant but does not state the detail asked for (a number, a deadline,
   an amount, who approves), say what the policy does say and state clearly that the detail
   is not specified. Set "covered" to "partial".
4. If no policy covers the question, set "policy_title" to null, "covered" to "no" and answer
   "This is not covered by the company policies."
5. Never invent numbers, deadlines, amounts, or rules. Never use outside knowledge.

Reply with JSON only: {"answer": string, "policy_title": string or null, "covered": "yes" | "partial" | "no"}"""


def _prompt(question: str, policies) -> str:
    lines = "\n".join(p.as_line() for p in policies)
    return f"POLICIES:\n{lines}\n\nQUESTION: {question}"


def _finish(method, question, data, usage, latency, retrieved=None, embed_tokens=0) -> Result:
    answer = str(data.get("answer", "")).strip()
    title = data.get("policy_title") or None
    covered = str(data.get("covered", "yes")).lower()
    label, reason, policy_text = grounding.check(answer, title, covered)
    return Result(method, question, answer, title, policy_text, latency,
                  usage["input_tokens"], usage["output_tokens"], usage["total_tokens"] + embed_tokens,
                  embed_tokens, covered, label, reason, retrieved or [])


# ---------- Approach 2: LLM without a vector index ----------
def answer_full_context(question: str) -> Result:
    t0 = time.perf_counter()
    data, usage = gemini.generate_json(SYSTEM_PROMPT, _prompt(question, load_policies()))
    return _finish("llm_full", question, data, usage, (time.perf_counter() - t0) * 1000)


# ---------- Approach 3: LLM with a vector index ----------
INDEX_FILE = ROOT / "data" / "policy_embeddings.npz"


def _doc_text(p) -> str:
    return f"{p.title}. Category: {p.category}. {p.text}"


@lru_cache(maxsize=1)
def vector_index() -> np.ndarray:
    """Embeds all policies once and caches them to disk (commit the .npz file to the repo)."""
    policies = load_policies()
    if INDEX_FILE.exists():
        f = np.load(INDEX_FILE, allow_pickle=False)
        if str(f["model"]) == config.EMBEDDING_MODEL and f["vectors"].shape[0] == len(policies):
            return f["vectors"]
    vecs = np.array(gemini.embed([_doc_text(p) for p in policies], "RETRIEVAL_DOCUMENT"), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.savez(INDEX_FILE, vectors=vecs, model=np.array(config.EMBEDDING_MODEL))
    return vecs


def retrieve(question: str, k: int) -> list[tuple[float, int]]:
    q = np.array(gemini.embed([question], "RETRIEVAL_QUERY")[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    sims = vector_index() @ q
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i)) for i in top]


def answer_rag(question: str, k: int | None = None) -> Result:
    k = k or config.RAG_TOP_K
    vector_index()  # build/load index outside the timed section (it is done once, offline)
    t0 = time.perf_counter()
    hits = retrieve(question, k)
    policies = load_policies()
    chosen = [policies[i] for _, i in hits]
    data, usage = gemini.generate_json(SYSTEM_PROMPT, _prompt(question, chosen))
    latency = (time.perf_counter() - t0) * 1000
    embed_tokens = max(1, len(question) // 4)  # the embedding API does not report usage; ~4 chars/token
    retrieved = [f"{policies[i].title} ({s:.3f})" for s, i in hits]
    return _finish("llm_rag", question, data, usage, latency, retrieved, embed_tokens)
