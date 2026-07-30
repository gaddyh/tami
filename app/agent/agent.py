import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import langsmith as ls
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.db.connection import get_checkpointer
from app.tools.reminders import save_reminder, set_chat_id


_MODEL = "gpt-5.4-mini"
_APP_VERSION = "v2"

_agent = None
_checkpointer = None

# Maps the real application thread ID to a LangGraph checkpoint thread.
# A new checkpoint thread is created after a reminder is completed.
_thread_id_map: dict[str, str] = {}


ReminderBehavior = Literal[
    "save_reminder",
    "clarify_time",
    "clarify_subject",
]


class ReminderAgentResponse(BaseModel):
    """Validated final output produced by the agent."""

    behavior: ReminderBehavior = Field(
        description=(
            "The behavioral decision made by the agent. "
            "Use save_reminder only when the save_reminder tool "
            "was actually called. "
            "Use clarify_time when the reminder subject is known "
            "but the due time is missing. "
            "Use clarify_subject when the due time is known "
            "but the reminder subject is missing."
        )
    )

    response: str = Field(
        description=(
            "The final user-facing response. It must be written "
            "in the same language as the user's message."
        )
    )


class ReminderRunResult(BaseModel):
    """Application-level result returned by run_agent."""

    response: str
    behavior: ReminderBehavior
    latency_seconds: float
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]


SYSTEM_PROMPT = (
    "You are a reminder assistant. Your job is to extract two pieces of "
    "information from the user's message: the reminder subject and the "
    "reminder due time. "
    "\n\n"
    "The current datetime is provided in a system message. Use it to resolve "
    "relative dates and times such as 'tomorrow', 'Friday', 'in 10 minutes', "
    "'in 2 hours', and similar expressions. "
    "\n\n"
    "If BOTH the subject and due time are stated or can be inferred from the "
    "user's message, call save_reminder immediately with:\n"
    "- subject: a string describing what the user should be reminded about\n"
    "- due_time: an ISO 8601 datetime string\n"
    "\n"
    "Relative expressions such as 'in 10 minutes', 'in 2 hours', "
    "'tomorrow at 9am', and 'Friday at 2pm' are complete due times. "
    "Calculate the exact datetime from the current datetime and call the tool. "
    "\n\n"
    "If the subject is genuinely missing because the user did not say what "
    "to remind them about, ask for the subject. "
    "\n"
    "If the due time is genuinely missing because the user provided no date, "
    "time of day, or relative time expression, ask for the due time. "
    "\n\n"
    "When the user supplies missing information in a follow-up message, "
    "combine it with the previous conversation context. Do not ask again for "
    "information that the user already supplied. Once both values are known, "
    "call save_reminder. "
    "\n\n"
    "After save_reminder succeeds, confirm the reminder to the user. "
    "Respond in the same language as the user. "
    "\n\n"
    "BEHAVIOR CONTRACT:\n"
    "Your final structured response must contain exactly one behavior:\n"
    "\n"
    "1. save_reminder\n"
    "   Use this only after save_reminder was actually called during the run.\n"
    "\n"
    "2. clarify_time\n"
    "   Use this when the reminder subject is known but the due time is "
    "missing.\n"
    "\n"
    "3. clarify_subject\n"
    "   Use this when the due time is known but the reminder subject is "
    "missing.\n"
    "\n"
    "Never report behavior=save_reminder unless save_reminder was actually "
    "called. "
    "\n\n"
    "EXAMPLES:\n"
    "\n"
    "User: תזכיר לי להתקשר לאמא מחר בשעה 5 אחר הצהריים\n"
    "Assistant action: call save_reminder with subject='call mom' and "
    "due_time=<tomorrow at 17:00>.\n"
    "Final behavior: save_reminder\n"
    "Final response: סבבה, אזכיר לך להתקשר לאמא מחר בשעה 17:00.\n"
    "\n"
    "User: תזכיר לי לקחת תרופות בעוד 30 דקות\n"
    "Assistant action: call save_reminder with subject='take medicine' and "
    "due_time=<current datetime plus 30 minutes>.\n"
    "Final behavior: save_reminder\n"
    "Final response: התראה נקבעה לקחת תרופות בעוד 30 דקות.\n"
    "\n"
    "User: תזכיר לי על הפגישה ביום שישי בשעה 2 אחר הצהריים\n"
    "Assistant action: call save_reminder with subject='meeting' and "
    "due_time=<Friday at 14:00>.\n"
    "Final behavior: save_reminder\n"
    "Final response: סבבה, אזכיר לך על הפגישה ביום שישי בשעה 14:00.\n"
    "\n"
    "User: תזכיר לי להתקשר לאבא מחר\n"
    "Final behavior: clarify_time\n"
    "Final response: באיזו שעה מחר לקבוע את התזכורת?\n"
    "\n"
    "User: תזכיר לי בעוד 10 דקות\n"
    "Final behavior: clarify_subject\n"
    "Final response: על מה להזכיר לך?\n"
    "\n"
    "User: תזכיר לי מחר בשעה 9 בבוקר\n"
    "Final behavior: clarify_subject\n"
    "Final response: על מה להזכיר לך?\n"
    "\n"
    "User follow-up: להתקשר לבנק\n"
    "Assistant action: combine the previous due time with the new subject and "
    "call save_reminder with subject='call the bank' and "
    "due_time=<tomorrow at 09:00>.\n"
    "Final behavior: save_reminder\n"
    "Final response: סבבה, אזכיר לך להתקשר לבנק מחר בשעה 09:00.\n"
    "\n"
    "User: עירית 40 דקות\n"
    "Assistant action: infer subject='עירית' and due_time=<current datetime "
    "plus 40 minutes>, then call save_reminder.\n"
    "Final behavior: save_reminder\n"
    "Final response: סבבה, אזכיר לך עירית בעוד 40 דקות."
)


def _get_agent():
    """Create and cache the reminder agent."""

    global _agent, _checkpointer

    if _agent is None:
        _checkpointer = get_checkpointer()

        _agent = create_agent(
            model=ChatOpenAI(
                model=_MODEL,
            ),
            tools=[save_reminder],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_checkpointer,
            response_format=ReminderAgentResponse,
        )

    return _agent


def _anonymous_thread_id(thread_id: str) -> str:
    """Create a stable anonymous identifier without tracing a phone number."""

    return hashlib.sha256(
        thread_id.encode("utf-8")
    ).hexdigest()[:16]


def _sanitize_run_agent_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve the user message for evaluation without tracing the raw
    WhatsApp phone number or business thread identifier.
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


def _extract_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract a flat list of tool calls from serialized messages."""

    return [
        tool_call
        for message in messages
        for tool_call in message.get("tool_calls", [])
    ]


def _sanitize_run_agent_outputs(
    outputs: ReminderRunResult,
) -> dict[str, Any]:
    """
    Store a compact evaluation-friendly parent span.

    The detailed LangGraph message and tool history remains available
    in the child trace.
    """

    called_tools = [
        tool_call["name"]
        for tool_call in outputs.tool_calls
    ]

    return {
        "response": outputs.response,
        "behavior": outputs.behavior,
        "latency_seconds": outputs.latency_seconds,
        "called_tools": called_tools,
        "message_count": len(outputs.messages),
        "reminder_saved": "save_reminder" in called_tools,
    }


def _serialize_messages(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Serialize LangChain messages for callers and offline evaluation."""

    result_messages: list[dict[str, Any]] = []

    for message in messages:
        serialized_message: dict[str, Any] = {
            "role": message.type,
            "content": message.content,
        }

        message_tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if message_tool_calls:
            serialized_message["tool_calls"] = [
                {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}),
                }
                for tool_call in message_tool_calls
            ]

        result_messages.append(serialized_message)

    return result_messages


def _validate_behavior_tool_consistency(
    behavior: ReminderBehavior,
    has_save_reminder_call: bool,
) -> None:
    """
    Ensure that the model-reported behavior agrees with observed execution.

    The structured response represents the model's declared decision.
    The tool trace represents what actually happened.
    """

    if (
        behavior == "save_reminder"
        and not has_save_reminder_call
    ):
        raise RuntimeError(
            "Agent reported behavior='save_reminder', "
            "but save_reminder was not called."
        )

    if (
        behavior != "save_reminder"
        and has_save_reminder_call
    ):
        raise RuntimeError(
            f"Agent reported behavior={behavior!r}, "
            "but save_reminder was called."
        )


@ls.traceable(
    name="run_reminder_agent",
    run_type="chain",
    tags=[
        "tami",
        "reminder-agent",
    ],
    metadata={
        "agent_type": "reminder",
        "model": _MODEL,
        "app_version": _APP_VERSION,
    },
    process_inputs=_sanitize_run_agent_inputs,
    process_outputs=_sanitize_run_agent_outputs,
)
def run_agent(
    user_message: str,
    thread_id: str,
    channel: str = "cli",
) -> ReminderRunResult:
    """
    Run one turn of the reminder agent.

    The caller-provided thread_id is used by the business logic but is
    replaced with an anonymized identifier in LangSmith traces.
    """

    agent = _get_agent()

    # The real chat ID is required by the reminder persistence tool.
    set_chat_id(thread_id)

    checkpoint_thread = _thread_id_map.get(
        thread_id,
        str(uuid.uuid4()),
    )

    _thread_id_map[thread_id] = checkpoint_thread

    anonymous_user_id = _anonymous_thread_id(
        thread_id
    )

    run = ls.get_current_run_tree()

    if run is not None:
        run.metadata.update(
            {
                "channel": channel,
                "anonymous_user_id": anonymous_user_id,
                "checkpoint_thread_id": checkpoint_thread,
            }
        )

    config = {
        "configurable": {
            "thread_id": checkpoint_thread,
        },
        "metadata": {
            "environment": "development",
            "app_version": _APP_VERSION,
            "channel": channel,
            "agent_type": "reminder",
            "model": _MODEL,
            "anonymous_user_id": anonymous_user_id,
        },
        "tags": [
            "tami",
            "reminder-agent",
            channel,
        ],
    }

    now = datetime.now(
        timezone.utc
    ).isoformat()

    messages = [
        SystemMessage(
            content=(
                "Current datetime (UTC): "
                f"{now}"
            )
        ),
        HumanMessage(
            content=user_message
        ),
    ]

    start = time.perf_counter()

    try:
        result = agent.invoke(
            {
                "messages": messages,
            },
            config=config,
        )
    except Exception as exc:
        if run is not None:
            run.metadata.update(
                {
                    "agent_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

        raise

    latency = time.perf_counter() - start

    result_messages = _serialize_messages(
        result["messages"]
    )

    tool_calls = _extract_tool_calls(
        result_messages
    )

    has_save = any(
        tool_call.get("name") == "save_reminder"
        for tool_call in tool_calls
    )

    structured_response = result.get(
        "structured_response"
    )

    if structured_response is None:
        raise RuntimeError(
            "Agent completed without returning "
            "a structured_response."
        )

    if not isinstance(
        structured_response,
        ReminderAgentResponse,
    ):
        structured_response = (
            ReminderAgentResponse.model_validate(
                structured_response
            )
        )

    behavior = structured_response.behavior

    _validate_behavior_tool_consistency(
        behavior=behavior,
        has_save_reminder_call=has_save,
    )

    if has_save:
        # A completed reminder closes the current conversational flow.
        # The next message from this user should begin with clean state.
        _thread_id_map[thread_id] = str(
            uuid.uuid4()
        )

    if run is not None:
        run.metadata.update(
            {
                "agent_status": "completed",
                "behavior": behavior,
                "latency_seconds": latency,
                "message_count": len(result_messages),
                "tool_call_count": len(tool_calls),
                "save_reminder_called": has_save,
                "checkpoint_reset": has_save,
            }
        )

    return ReminderRunResult(
        response=structured_response.response,
        behavior=behavior,
        latency_seconds=latency,
        messages=result_messages,
        tool_calls=tool_calls,
    )