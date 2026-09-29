"""Approach 1: rules-based search (BM25 keyword ranking + synonym rules). No LLM, no tokens.

The answer is always the policy text verbatim, so it can never invent facts, but it can
pick the wrong policy or miss paraphrased questions.
"""
import math
import re
import time
from collections import Counter
from functools import lru_cache

from .policies import load_policies
from .result import Result

STOPWORDS = set("""
a about after all am an and any are as at be been before being but by can could do does doing
for from get got have has how i if in into is it its me my of on or our should so than that the
their them then there these they this to up us was we what when where which who why will with
would you your policy policies company allowed allow ok okay need needs must may much many long
""".split())

# Rule table: words people use -> words the policies use.
SYNONYMS = {
    "wfh": ["remote"], "home": ["remote"], "homeoffice": ["remote", "home", "office"],
    "holiday": ["vacation", "holiday"], "holidays": ["vacation", "holiday"], "pto": ["vacation"],
    "leave": ["leave", "vacation"], "time": ["leave"], "off": ["leave", "vacation"],
    "sick": ["sick", "illness"], "ill": ["sick", "illness"],
    "baby": ["parental"], "maternity": ["parental"], "paternity": ["parental"],
    "died": ["bereavement"], "death": ["bereavement"], "funeral": ["bereavement"], "passed": ["bereavement"],
    "abroad": ["abroad"], "overseas": ["abroad"], "country": ["abroad"],
    "hacked": ["security", "incident"], "virus": ["security", "incident"], "phishing": ["security", "incident"],
    "breach": ["security", "incident"],
    "wear": ["dress", "attire"], "clothes": ["dress", "attire"], "outfit": ["dress", "attire"],
    "receipt": ["receipts", "expenses"], "trip": ["travel"], "flight": ["travel"], "hotel": ["travel"],
    "journalist": ["media", "press"], "press": ["press", "media"], "reporter": ["media"],
    "course": ["training", "tuition"], "certification": ["training"], "degree": ["tuition"],
    "education": ["tuition", "training"],
    "password": ["passwords"], "login": ["passwords"],
    "hours": ["hours"], "schedule": ["hours"], "start": ["hours"],
    "salary": ["promotion"], "raise": ["promotion"],
    "car": ["vehicle"], "parking": ["parking"],
    "phone": ["mobile", "device"], "laptop": ["laptop", "hardware"], "computer": ["laptop", "hardware"],
    "buy": ["purchases", "procurement"], "purchase": ["purchases", "procurement"],
    "spend": ["expenses"], "expense": ["expenses"], "cost": ["expenses"],
    "blog": ["content", "publishing"], "post": ["publishing", "social"], "linkedin": ["social", "media"],
    "twitter": ["social", "media"], "harass": ["harassment"], "bullying": ["harassment"],
    "report": ["report"], "problem": ["issues", "helpdesk"], "broken": ["issues", "helpdesk"],
    "it": ["it"], "new": ["new"], "hire": ["hires"], "newbie": ["new", "hires"],
    "trial": ["probation"], "probation": ["probation"],
    "return": ["return", "returns"], "refund": ["returns"], "warranty": ["warranty"],
    "quit": ["departing", "exit"], "leaving": ["departing", "exit"], "resign": ["departing", "exit"],
    "vpn": ["vpn", "remote", "access"], "badge": ["badges", "access"],
}

TITLE_WEIGHT = 2  # title words count double
K1, B = 1.5, 0.75
MIN_SCORE = 2.5   # below this the search says "no matching policy"


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9$]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def expand_query(question: str) -> list[str]:
    raw = re.findall(r"[a-z0-9$]+", question.lower())
    terms = []
    for w in raw:
        if w in SYNONYMS:
            terms += [_stem(s) for s in SYNONYMS[w]]
        if w not in STOPWORDS:
            terms.append(_stem(w))
    return terms


@lru_cache(maxsize=1)
def _index():
    docs = []
    for p in load_policies():
        docs.append(tokenize(p.title) * TITLE_WEIGHT + tokenize(p.text) + tokenize(p.category))
    df = Counter(t for d in docs for t in set(d))
    avgdl = sum(len(d) for d in docs) / len(docs)
    return docs, df, avgdl


def search(question: str, k: int = 3) -> list[tuple[float, int]]:
    docs, df, avgdl = _index()
    n = len(docs)
    q = Counter(expand_query(question))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (K1 + 1) / (tf[term] + K1 * (1 - B + B * len(d) / avgdl))
        scores.append((s, i))
    scores.sort(reverse=True)
    return scores[:k]


def answer(question: str) -> Result:
    _index()  # built once at startup, not part of the per-question time
    t0 = time.perf_counter()
    policies = load_policies()
    hits = search(question)
    latency = (time.perf_counter() - t0) * 1000
    retrieved = [f"{policies[i].title} ({s:.2f})" for s, i in hits]
    best_score, best = hits[0]
    if best_score < MIN_SCORE:
        return Result("rules", question, "No matching policy found.", None, None, latency,
                      covered="no", grounding="abstained", grounding_reason="Score below threshold",
                      retrieved=retrieved)
    p = policies[best]
    return Result("rules", question, f"According to the {p.title}: {p.text}", p.title, p.text, latency,
                  covered="yes", grounding="grounded",
                  grounding_reason="Returns policy text verbatim", retrieved=retrieved)
