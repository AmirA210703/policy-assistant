"""Automatic check of whether an LLM answer is supported by the policy it cites.

Used live on the website (cheap, no extra API call). The evaluation script adds a stricter
LLM-as-judge check on top of this.
"""
import re

from .policies import find_policy

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "ninety": "90",
}


def numbers_in(text: str) -> set[str]:
    text = text.lower().replace(",", "")
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    for w, n in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            nums.add(n)
    return nums


def check(answer: str, policy_title: str | None, covered: str) -> tuple[str, str, str | None]:
    """Returns (label, reason, policy_text)."""
    if not policy_title and covered != "no":
        return "unsupported", "Gives an answer without citing any policy", None
    if not policy_title:
        return "abstained", "No policy cited; the model says it is not covered", None
    policy = find_policy(policy_title)
    if policy is None:
        return "unsupported", f"Cites a policy that does not exist: '{policy_title}'", None
    extra = numbers_in(answer) - numbers_in(policy.text + " " + policy.title)
    if extra:
        return "unsupported", f"Answer contains numbers not in the policy: {', '.join(sorted(extra))}", policy.text
    return "grounded", "Cited policy exists and no unstated numbers", policy.text
