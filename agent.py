import time
import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from db import get_checkpointer
from tools import save_reminder, set_chat_id

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
    "Assistant: [calls save_reminder(subject=\"call mom\", due_time=<tomorrow 17:00>)]\n"
    "  -> סבבה, אזכיר לך להתקשר לאמא מחר בשעה 17:00.\n\n"
    "User: תזכיר לי לקחת תרופות בעוד 30 דקות\n"
    "Assistant: [calls save_reminder(subject=\"take medicine\", due_time=<now + 30min>)]\n"
    "  -> התראה נקבעה לקחת תרופות בעוד 30 דקות.\n\n"
    "User: תזכיר לי על הפגישה ביום שישי בשעה 2 אחר הצהריים\n"
    "Assistant: [calls save_reminder(subject=\"meeting\", due_time=<Friday 14:00>)]\n"
    "  -> סבבה, אזכיר לך על הפגישה ביום שישי בשעה 14:00.\n\n"
    "User: תזכיר לי להתקשר לאבא מחר\n"
    "Assistant: באיזו שעה מחר לקבוע את התזכורת?\n\n"
    "User: תזכיר לי בעוד 10 דקות\n"
    "Assistant: על מה להזכיר לך?\n\n"
    "User: תזכיר לי מחר בשעה 9 בבוקר\n"
    "Assistant: על מה להזכיר לך?\n"
    "User: להתקשר לבנק\n"
    "Assistant: [calls save_reminder(subject=\"call the bank\", due_time=<tomorrow 09:00>)]\n"
    "  -> סבבה, אזכיר לך להתקשר לבנק מחר בשעה 09:00."
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


def run_agent(
    user_message: str,
    thread_id: str,
) -> tuple[str, float, list[dict]]:
    agent = _get_agent()
    set_chat_id(thread_id)

    checkpoint_thread = _thread_id_map.get(thread_id, str(uuid.uuid4()))
    _thread_id_map[thread_id] = checkpoint_thread
    config = {"configurable": {"thread_id": checkpoint_thread}}

    now = datetime.now(timezone.utc).isoformat()
    messages = [
        SystemMessage(content=f"Current datetime (UTC): {now}"),
        HumanMessage(content=user_message),
    ]

    start = time.perf_counter()
    result = agent.invoke({"messages": messages}, config=config)
    latency = time.perf_counter() - start

    result_messages = []
    for m in result["messages"]:
        msg = {"role": m.type, "content": m.content}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"]}
                for tc in tool_calls
            ]
        result_messages.append(msg)

    has_save = any(
        tc.get("name") == "save_reminder"
        for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
    )
    if has_save:
        _thread_id_map[thread_id] = str(uuid.uuid4())

    return result["messages"][-1].content, latency, result_messages
