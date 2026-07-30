import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import langsmith as ls
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.db.connection import get_checkpointer
from app.tools.reminders import save_reminder, set_chat_id

_MODEL = "gpt-5.4-mini"

_agent = None
_checkpointer = None

_thread_id_map: dict[str, str] = {}


SYSTEM_PROMPT = (
    "You are a reminder assistant. Your job is to extract two pieces of "
    "information from the user's message: the reminder subject and the "
    "reminder due time. "
    "The current datetime is provided in the system message — use it to "
    "resolve relative dates like 'tomorrow', 'Friday', 'in 10 minutes', etc. "
    "If BOTH the subject and due time are stated or can be inferred from the "
    "user's message, call save_reminder immediately with the subject (string) "
    "and the due_time as an ISO 8601 datetime string. "
    "Relative time expressions like 'in 10 minutes', 'in 2 hours', 'tomorrow "
    "at 9am', 'Friday at 2pm' are ALL valid complete due times — compute the "
    "exact datetime from the current datetime and call the tool. "
    "If the subject is genuinely missing (the user didn't say what to remind "
    "them about), ask for it. If the time is genuinely missing (no time of "
    "day or relative time given at all), ask for it. "
    "When the user provides the missing information in a follow-up message, "
    "combine it with the previous conversation context and call save_reminder "
    "with the complete information — do NOT ask for information that was "
    "already provided in earlier messages. "
    "After saving, confirm the reminder to the user.\n\n"
    "Examples:\n"
    "User: תזכיר לי להתקשר לאמא מחר בשעה 5 אחר הצהריים\n"
    "Assistant: [calls save_reminder(subject=\"call mom\", "
    "due_time=<tomorrow 17:00>)]\n"
    "  -> סבבה, אזכיר לך להתקשר לאמא מחר בשעה 17:00.\n\n"
    "User: תזכיר לי לקחת תרופות בעוד 30 דקות\n"
    "Assistant: [calls save_reminder(subject=\"take medicine\", "
    "due_time=<now + 30min>)]\n"
    "  -> התראה נקבעה לקחת תרופות בעוד 30 דקות.\n\n"
    "User: תזכיר לי על הפגישה ביום שישי בשעה 2 אחר הצהריים\n"
    "Assistant: [calls save_reminder(subject=\"meeting\", "
    "due_time=<Friday 14:00>)]\n"
    "  -> סבבה, אזכיר לך על הפגישה ביום שישי בשעה 14:00.\n\n"
    "User: תזכיר לי להתקשר לאבא מחר\n"
    "Assistant: באיזו שעה מחר לקבוע את התזכורת?\n\n"
    "User: תזכיר לי בעוד 10 דקות\n"
    "Assistant: על מה להזכיר לך?\n\n"
    "User: תזכיר לי מחר בשעה 9 בבוקר\n"
    "Assistant: על מה להזכיר לך?\n"
    "User: להתקשר לבנק\n"
    "Assistant: [calls save_reminder(subject=\"call the bank\", "
    "due_time=<tomorrow 09:00>)]\n"
    "  -> סבבה, אזכיר לך להתקשר לבנק מחר בשעה 09:00.\n\n"
    "User: עירית 40 דקות\n"
    "Assistant: [calls save_reminder(subject=\"עירית\", "
    "due_time=<now + 40min>)]\n"
    "  -> סבבה, אזכיר לך עירית בעוד 40 דקות."
)


def _get_agent():
    global _agent, _checkpointer

    if _agent is None:
        _checkpointer = get_checkpointer()

        _agent = create_react_agent(
            model=ChatOpenAI(model=_MODEL),
            tools=[save_reminder],
            prompt=SYSTEM_PROMPT,
            checkpointer=_checkpointer,
        )

    return _agent


def _anonymous_thread_id(thread_id: str) -> str:
    """Create a stable identifier without tracing a phone number."""
    return hashlib.sha256(
        thread_id.encode("utf-8")
    ).hexdigest()[:16]


def _sanitize_run_agent_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve the user message for evaluation, but do not trace the raw
    WhatsApp phone number used as thread_id.
    """
    thread_id = str(inputs.get("thread_id", ""))

    return {
        "user_message": inputs.get("user_message"),
        "anonymous_user_id": (
            _anonymous_thread_id(thread_id)
            if thread_id
            else None
        ),
        "channel": inputs.get("channel"),
    }


def _sanitize_run_agent_outputs(
    outputs: tuple[str, float, list[dict]],
) -> dict[str, Any]:
    """
    Avoid duplicating the complete LangGraph message history on the
    run_reminder_agent span. The detailed history remains available
    inside the LangGraph child trace.
    """
    response_text, latency, result_messages = outputs

    called_tools = [
        tool_call["name"]
        for message in result_messages
        for tool_call in message.get("tool_calls", [])
    ]

    return {
        "response": response_text,
        "latency_seconds": latency,
        "called_tools": called_tools,
        "message_count": len(result_messages),
        "reminder_saved": "save_reminder" in called_tools,
    }


@ls.traceable(
    name="run_reminder_agent",
    run_type="chain",
    tags=["tami", "reminder-agent"],
    metadata={
        "agent_type": "reminder",
        "model": _MODEL,
        "app_version": "v1",
    },
    process_inputs=_sanitize_run_agent_inputs,
    process_outputs=_sanitize_run_agent_outputs,
)
def run_agent(
    user_message: str,
    thread_id: str,
    channel: str = "cli",
) -> tuple[str, float, list[dict]]:
    agent = _get_agent()

    # The real ID is required by the business logic and checkpointer,
    # but is removed from the traced inputs by process_inputs.
    set_chat_id(thread_id)

    checkpoint_thread = _thread_id_map.get(
        thread_id,
        str(uuid.uuid4()),
    )
    _thread_id_map[thread_id] = checkpoint_thread

    run = ls.get_current_run_tree()

    if run is not None:
        run.metadata.update(
            {
                "channel": channel,
                "anonymous_user_id": _anonymous_thread_id(
                    thread_id
                ),
                "checkpoint_thread_id": checkpoint_thread,
            }
        )

    config = {
        "configurable": {
            "thread_id": checkpoint_thread,
        },
        "metadata": {
            "environment": "development",
            "app_version": "v1",
            "channel": channel,
            "agent_type": "reminder",
            "model": _MODEL,
            "anonymous_user_id": _anonymous_thread_id(
                thread_id
            ),
        },
        "tags": [
            "tami",
            "reminder-agent",
            channel,
        ],
    }

    now = datetime.now(timezone.utc).isoformat()

    messages = [
        SystemMessage(
            content=f"Current datetime (UTC): {now}"
        ),
        HumanMessage(content=user_message),
    ]

    start = time.perf_counter()

    try:
        result = agent.invoke(
            {"messages": messages},
            config=config,
        )
    except Exception as exc:
        if run is not None:
            run.metadata.update(
                {
                    "agent_status": "failed",
                    "error_type": type(exc).__name__,
                }
            )

        raise

    latency = time.perf_counter() - start

    result_messages: list[dict[str, Any]] = []

    for message in result["messages"]:
        serialized_message: dict[str, Any] = {
            "role": message.type,
            "content": message.content,
        }

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if tool_calls:
            serialized_message["tool_calls"] = [
                {
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                }
                for tool_call in tool_calls
            ]

        result_messages.append(serialized_message)

    has_save = any(
        tool_call.get("name") == "save_reminder"
        for message in result["messages"]
        for tool_call in (
            getattr(message, "tool_calls", None) or []
        )
    )

    if has_save:
        # Start a new conversational checkpoint after a completed
        # reminder so a future reminder does not inherit stale state.
        _thread_id_map[thread_id] = str(uuid.uuid4())

    final_response = result["messages"][-1].content

    if run is not None:
        run.metadata.update(
            {
                "agent_status": "completed",
                "latency_seconds": latency,
                "message_count": len(result_messages),
                "save_reminder_called": has_save,
                "checkpoint_reset": has_save,
            }
        )

    return (
        final_response,
        latency,
        result_messages,
    )