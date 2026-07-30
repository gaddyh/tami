from __future__ import annotations

import uuid
from typing import Any

from app.agent.agent import run_agent


def extract_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    for message in messages:
        for call in message.get("tool_calls", []):
            calls.append(
                {
                    "name": call["name"],
                    "args": call.get("args", {}),
                }
            )

    return calls


def infer_behavior(
    response: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    called_tools = {
        call["name"]
        for call in tool_calls
    }

    if "save_reminder" in called_tools:
        return "save_reminder"

    normalized = response.casefold()

    time_signals = [
        "when",
        "what time",
        "which time",
        "מתי",
        "באיזו שעה",
    ]

    subject_signals = [
        "what should",
        "what would you like",
        "remind you about",
        "מה להזכיר",
        "על מה",
    ]

    if any(signal in normalized for signal in time_signals):
        return "clarify_time"

    if any(signal in normalized for signal in subject_signals):
        return "clarify_subject"

    return "unknown"


def run_reminder_target(inputs: dict) -> dict:
    thread_id = f"langsmith-eval-{uuid.uuid4()}"

    result = run_agent(
        user_message=inputs["user_message"],
        thread_id=thread_id,
        channel="langsmith-evaluation",
    )

    return {
        "behavior": result.behavior,
        "response": result.response,
        "tool_calls": result.tool_calls,
        "called_tools": [
            call["name"]
            for call in result.tool_calls
        ],
        "latency_seconds": result.latency_seconds,
    }