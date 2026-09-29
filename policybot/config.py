"""Settings read from environment variables, a local .env file, or Streamlit secrets."""
import os

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit Community Cloud stores secrets here
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Model used by evaluate.py to judge answers. A different model has its own daily quota.
JUDGE_MODEL = get_setting("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = get_setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
# 0 turns off Gemini "thinking" so latency/token numbers stay comparable. Use "" to leave it on.
THINKING_BUDGET = get_setting("GEMINI_THINKING_BUDGET", "0")
RAG_TOP_K = int(get_setting("RAG_TOP_K", "4"))
STUDENT_NAME = get_setting("STUDENT_NAME", "Student name")
