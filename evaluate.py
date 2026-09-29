"""Runs every test question through all three approaches and writes results/.

    python evaluate.py              # all questions, all methods
    python evaluate.py --no-judge   # skip the LLM-as-judge step (saves API calls)
    python evaluate.py --sleep 4    # pause between calls (Gemini free tier rate limits)

Outputs:
    results/eval_results.csv   one row per question x method
    results/summary.json       the two tables shown on the website
"""
import argparse
import csv
import json
import time
from pathlib import Path

from policybot import gemini
from policybot.engine import METHODS, answer
from policybot.policies import find_policy

OUT = Path(__file__).parent / "results"

JUDGE_PROMPT = """You check whether a policy assistant's answer is fully supported by the policy it cites.
An answer is SUPPORTED if every factual claim in it (numbers, deadlines, amounts, permissions,
who approves) is stated in the policy text, or if it correctly says the detail is not specified.
It is UNSUPPORTED if it adds any fact, number or rule that is not in the policy text.
If no policy is cited and the answer only says the topic is not covered, it is SUPPORTED.

Reply with JSON only: {"supported": true | false, "reason": string}"""


def judge(question: str, answer_text: str, policy_title: str | None) -> tuple[bool, str]:
    p = find_policy(policy_title)
    policy_block = f"[{p.title}] {p.text}" if p else "(no policy cited)"
    prompt = f"QUESTION: {question}\nCITED POLICY: {policy_block}\nANSWER: {answer_text}"
    data, _ = gemini.generate_json(JUDGE_PROMPT, prompt)
    return bool(data.get("supported")), str(data.get("reason", ""))


def is_correct(row: dict, r) -> bool:
    expected = row["expected_policy"].split("|")
    if expected == ["NONE"]:
        return r.policy_title is None
    p = find_policy(r.policy_title)
    return p is not None and p.title in expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    questions = list(csv.DictReader(open("data/test_questions.csv", encoding="utf-8")))
    methods = args.methods.split(",")
    rows = []
    for q in questions:
        for m in methods:
            r = answer(q["question"], m)
            if args.no_judge or r.error:
                supported, reason = r.grounding in ("grounded", "abstained"), "automatic check only"
            elif m == "rules":
                supported, reason = True, "verbatim policy text"
            else:
                supported, reason = judge(q["question"], r.answer, r.policy_title)
                if args.sleep:
                    time.sleep(args.sleep)
            rows.append({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "expected_policy": q["expected_policy"], "method": m,
                "answer": r.answer, "policy_title": r.policy_title or "NONE",
                "correct_policy": is_correct(q, r), "answered": r.policy_title is not None,
                "supported": supported, "judge_reason": reason,
                "auto_grounding": r.grounding, "latency_ms": round(r.latency_ms, 1),
                "total_tokens": r.total_tokens, "error": r.error or "",
            })
            print(f"[{q['id']:>2}] {m:9} correct={rows[-1]['correct_policy']!s:5} "
                  f"supported={supported!s:5} {r.latency_ms:7.0f} ms {r.total_tokens:5} tok  {r.policy_title}")
            if args.sleep and m != "rules":
                time.sleep(args.sleep)

    with open(OUT / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    json.dump(summarise(rows, methods), open(OUT / "summary.json", "w"), indent=2)
    print(f"\nWrote {OUT/'eval_results.csv'} and {OUT/'summary.json'}")


def summarise(rows, methods):
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    overall, by_type = [], []
    types = list(dict.fromkeys(r["type"] for r in rows))
    for m in methods:
        mr = [r for r in rows if r["method"] == m and not r["error"]]
        answered = [r for r in mr if r["answered"]]
        oos = [r for r in mr if r["type"] == "out_of_scope"]
        overall.append({
            "Approach": METHODS[m][0],
            "Correct policy (%)": pct([r["correct_policy"] for r in mr]),
            "Unsupported answers (%)": pct([not r["supported"] for r in answered]),
            "Out-of-scope refused (%)": pct([not r["answered"] for r in oos]),
            "Avg. response time (ms)": round(sum(r["latency_ms"] for r in mr) / len(mr), 1) if mr else None,
            "Avg. tokens / question": round(sum(r["total_tokens"] for r in mr) / len(mr)) if mr else None,
        })
        row = {"Approach": METHODS[m][0]}
        for t in types:
            row[t.replace("_", " ")] = pct([r["correct_policy"] and r["supported"] for r in mr if r["type"] == t])
        by_type.append(row)
    return {"overall": overall, "by_type": by_type,
            "note": "By question type: % of questions with the correct policy AND a supported answer."}


if __name__ == "__main__":
    main()
