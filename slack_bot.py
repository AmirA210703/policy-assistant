"""Slack bot (Socket Mode, runs on your own computer - no public URL needed).

    python slack_bot.py

Mention the bot in a channel:  @PolicyBot How many vacation days do I get?
or send it a direct message.
"""
import logging
import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from policybot import config
from policybot.engine import METHODS, answer

logging.basicConfig(level=logging.INFO)
METHOD = config.get_setting("BOT_METHOD", "llm_rag")
app = App(token=config.get_setting("SLACK_BOT_TOKEN"))


def format_reply(question: str) -> str:
    r = answer(question, METHOD)
    lines = [r.answer, ""]
    if r.policy_title:
        lines.append(f"*Relevant policy:* {r.policy_title}")
        if r.policy_text:
            lines.append(f"> {r.policy_text}")
    else:
        lines.append("*Relevant policy:* none found")
    lines.append(f"_{METHODS[METHOD][0]} · {r.latency_ms:.0f} ms · {r.total_tokens} tokens_")
    return "\n".join(lines)


@app.event("app_mention")
def on_mention(event, say):
    question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
    thread = event.get("thread_ts", event["ts"])
    if not question:
        say(text="Ask me a question about company policy, e.g. _How many vacation days do I get?_",
            thread_ts=thread)
        return
    say(text=format_reply(question), thread_ts=thread)


@app.event("message")
def on_dm(event, say):
    if event.get("channel_type") == "im" and not event.get("bot_id") and event.get("text"):
        say(text=format_reply(event["text"].strip()))


if __name__ == "__main__":
    SocketModeHandler(app, config.get_setting("SLACK_APP_TOKEN") or os.environ["SLACK_APP_TOKEN"]).start()
