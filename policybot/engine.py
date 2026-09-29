"""Single entry point used by the website, the Slack bot and the evaluation script."""
import time

from . import llm, rules
from .result import Result

METHODS = {
    "rules": ("Rules-based search", rules.answer),
    "llm_full": ("LLM without vector index", llm.answer_full_context),
    "llm_rag": ("LLM with vector index", llm.answer_rag),
}


def answer(question: str, method: str) -> Result:
    label, fn = METHODS[method]
    t0 = time.perf_counter()
    try:
        return fn(question)
    except Exception as e:
        return Result(method, question, f"Error: {e}", None, None, (time.perf_counter() - t0) * 1000,
                      grounding="error", grounding_reason=str(e), error=str(e))


def answer_all(question: str) -> dict[str, Result]:
    return {m: answer(question, m) for m in METHODS}
