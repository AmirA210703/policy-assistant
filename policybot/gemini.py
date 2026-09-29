"""Thin wrapper around the Gemini API (google-genai SDK) with token counting and retries."""
import json
import re
import time
from functools import lru_cache

from google import genai
from google.genai import types

from . import config

_budget_supported = True  # set to False once the model rejects a thinking budget


@lru_cache(maxsize=1)
def client() -> genai.Client:
    key = config.get_setting("GEMINI_API_KEY") or config.get_setting("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
    # 45 s timeout so a slow call fails visibly instead of hanging the website
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=45_000))


class DailyQuotaExceeded(RuntimeError):
    """The free-tier requests-per-day limit is used up; retrying today will not help."""


def _retry(fn, attempts: int = 5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits on the free tier -> back off and retry
            msg = str(e)
            print(f"[gemini] attempt {i + 1} failed: {msg[:200]}", flush=True)
            if "PerDay" in msg:
                raise DailyQuotaExceeded(msg) from e
            if i < attempts - 1 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg):
                time.sleep(3 * 2 ** i)  # 3, 6, 12, 24 s
                continue
            raise


def _config(system: str, use_thinking_budget: bool) -> types.GenerateContentConfig:
    kwargs = dict(system_instruction=system, temperature=0, response_mime_type="application/json",
                  max_output_tokens=1024,
                  automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    if use_thinking_budget and config.THINKING_BUDGET not in (None, ""):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(config.THINKING_BUDGET))
    return types.GenerateContentConfig(**kwargs)


def generate_json(system: str, prompt: str, model: str | None = None) -> tuple[dict, dict]:
    """Returns (parsed JSON, usage dict)."""
    model = model or config.GEMINI_MODEL

    def call(use_budget: bool):
        return client().models.generate_content(
            model=model, contents=prompt, config=_config(system, use_budget))

    global _budget_supported
    try:
        resp = _retry(lambda: call(_budget_supported))
    except DailyQuotaExceeded:
        raise
    except Exception as e:  # some models don't accept a thinking budget -> stop sending it
        msg = str(e)
        if not _budget_supported or not ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
            raise
        print("[gemini] retrying without thinking_budget", flush=True)
        _budget_supported = False
        resp = _retry(lambda: call(False))

    print(f"[gemini] {model} ok, thinking_budget={'on' if _budget_supported else 'off'}", flush=True)
    u = resp.usage_metadata
    usage = {
        "input_tokens": u.prompt_token_count or 0,
        "output_tokens": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
        "total_tokens": u.total_token_count or 0,
    }
    return parse_json(resp.text or ""), usage


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"answer": text, "policy_title": None, "covered": "no"}


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 50):  # batch to stay within request limits
        batch = texts[i:i + 50]
        resp = _retry(lambda: client().models.embed_content(
            model=config.EMBEDDING_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task_type)))
        out += [e.values for e in resp.embeddings]
    return out
