"""Runs every test question through all three approaches and writes results/.

    python evaluate.py              # all questions, all methods
    python evaluate.py --no-judge   # skip the LLM-as-judge step (saves API calls)
    python evaluate.py --sleep 4    # pause between calls (Gemini free tier rate limits)
    python evaluate.py --fresh      # start over instead of resuming

Progress is saved after every answer, so if the daily quota runs out you can simply run the
same command again the next day (quota resets at midnight Pacific time) and it continues.

Outputs:
    results/eval_results.csv   one row per question x method
    results/summary.json       the two tables shown on the website
"""
import argparse
import statistics
import csv
import json
import time
from pathlib import Path

from policybot import config, gemini
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
    data, _ = gemini.generate_json(JUDGE_PROMPT, prompt, model=config.JUDGE_MODEL)
    return bool(data.get("supported")), str(data.get("reason", ""))


def is_correct(row: dict, r) -> bool:
    expected = row["expected_policy"].split("|")
    if expected == ["NONE"]:
        return r.policy_title is None
    p = find_policy(r.policy_title)
    return p is not None and p.title in expected


FIELDS = ["id", "type", "question", "expected_policy", "method", "answer", "policy_title",
          "correct_policy", "answered", "supported", "judge_reason", "auto_grounding",
          "latency_ms", "total_tokens", "error"]


def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:  # CSV stores everything as text
        for k in ("correct_policy", "answered", "supported"):
            r[k] = r[k] == "True"
        r["latency_ms"] = float(r["latency_ms"])
        r["total_tokens"] = int(r["total_tokens"])
    return [r for r in rows if not r["error"]]  # failed rows are retried


def save(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary.json from the CSV")
    args = ap.parse_args()
    if args.summary_only:
        rows = load_done(OUT / "eval_results.csv")
        json.dump(summarise(rows, args.methods.split(",")), open(OUT / "summary.json", "w"), indent=2)
        print("Rebuilt", OUT / "summary.json")
        return

    OUT.mkdir(exist_ok=True)
    out_csv = OUT / "eval_results.csv"
    questions = list(csv.DictReader(open("data/test_questions.csv", encoding="utf-8")))
    methods = args.methods.split(",")
    rows = [] if args.fresh else load_done(out_csv)
    done = {(r["id"], r["method"]) for r in rows}
    if done:
        print(f"Resuming: {len(done)} answers already done.")
    print(f"Answer model: {config.GEMINI_MODEL} | judge model: {config.JUDGE_MODEL}\n")

    stopped = None
    for q in questions:
        for m in methods:
            if (q["id"], m) in done:
                continue
            r = answer(q["question"], m)
            if r.error and "PerDay" in r.error:
                stopped = f"daily quota for {config.GEMINI_MODEL}"
                break
            supported, reason = r.grounding in ("grounded", "abstained"), "automatic check only"
            if r.error:
                reason = "error"
            elif m == "rules":
                supported, reason = True, "verbatim policy text"
            elif not args.no_judge:
                try:
                    supported, reason = judge(q["question"], r.answer, r.policy_title)
                except gemini.DailyQuotaExceeded:
                    stopped = f"daily quota for judge model {config.JUDGE_MODEL}"
                    break
                except Exception as e:  # e.g. 503 overloaded: skip now, retried on next run
                    print(f"[{q['id']:>2}] {m:9} judge failed, will retry on next run: {str(e)[:80]}")
                    continue
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
            save(out_csv, rows)
            print(f"[{q['id']:>2}] {m:9} correct={rows[-1]['correct_policy']!s:5} "
                  f"supported={supported!s:5} {r.latency_ms:7.0f} ms {r.total_tokens:5} tok  {r.policy_title}")
            if args.sleep and m != "rules":
                time.sleep(args.sleep)
        if stopped:
            break

    total = len(questions) * len(methods)
    ok = len([r for r in rows if not r["error"]])
    if stopped:
        print(f"\nStopped: {stopped} is used up. {ok}/{total} answers saved.")
        print("Run the same command again after 09:00 Danish time to continue.")
    elif ok < total:
        print(f"\n{ok}/{total} answers saved. Run the same command again to fill in the rest.")
    else:
        print(f"\nDone: {ok}/{total} answers without errors.")
    json.dump(summarise(rows, methods), open(OUT / "summary.json", "w"), indent=2)
    print(f"Wrote {out_csv} and {OUT / 'summary.json'}")


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
            "Median response time (ms)": round(statistics.median(r["latency_ms"] for r in mr), 1) if mr else None,
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
