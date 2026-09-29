"""Loads the policy database (data/company_policies.csv)."""
import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_CSV = ROOT / "data" / "company_policies.csv"


@dataclass(frozen=True)
class Policy:
    id: int
    title: str
    department: str
    text: str
    category: str

    def as_line(self) -> str:
        return f"[{self.title}] ({self.category}) {self.text}"


@lru_cache(maxsize=1)
def load_policies() -> tuple[Policy, ...]:
    with open(POLICY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Policy(i, r["title"].strip(), r["department"].strip(), r["policy_text"].strip(), r["category"].strip())
        for i, r in enumerate(rows)
    )


def _norm(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", title.lower())
    t = re.sub(r"\bpolicy\b", " ", t)
    return " ".join(t.split())


def find_policy(title: str | None) -> Policy | None:
    """Match a title returned by an LLM to a real policy (case/punctuation-insensitive)."""
    if not title:
        return None
    wanted = _norm(title)
    for p in load_policies():
        if _norm(p.title) == wanted:
            return p
    return None
