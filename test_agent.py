import os
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv

from agent import run_agent
from run_tracker import record_test_latency

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

TOLERANCE = timedelta(hours=2)


def _print_trajectory(label: str, result_messages: list[dict]) -> None:
    print(f"\n--- {label} trajectory ---")
    for i, msg in enumerate(result_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        parts = []
        if content:
            parts.append(str(content))
        for tc in tool_calls:
            parts.append(f"[tool_call] {tc['name']}({tc['args']})")
        print(f"  [{i}] {role}: {' | '.join(parts) if parts else '(empty)'}")
    print(f"--- end {label} trajectory ---\n")


def _extract_tool_calls(result_messages: list[dict]) -> list[dict]:
    calls = []
    for msg in result_messages:
        if msg.get("role") == "ai":
            for tc in msg.get("tool_calls", []):
                calls.append({"name": tc["name"], "args": tc["args"]})
    return calls


def _assert_save_reminder_called(result_messages: list[dict]) -> dict:
    calls = _extract_tool_calls(result_messages)
    save_calls = [c for c in calls if c["name"] == "save_reminder"]
    assert len(save_calls) > 0, "save_reminder was not called"
    return save_calls[-1]["args"]


def _assert_save_reminder_not_called(result_messages: list[dict]) -> None:
    calls = _extract_tool_calls(result_messages)
    save_calls = [c for c in calls if c["name"] == "save_reminder"]
    assert len(save_calls) == 0, f"save_reminder was called unexpectedly: {save_calls}"


def _check_subject(actual: str, expected_keywords: list[str]) -> None:
    actual_lower = actual.lower()
    for kw in expected_keywords:
        assert kw.lower() in actual_lower, (
            f"Expected subject to contain '{kw}', got '{actual}'"
        )


def _check_due_time(actual: str, expected: datetime, tolerance: timedelta = TOLERANCE) -> None:
    parsed = datetime.fromisoformat(actual)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    diff = abs(parsed - expected)
    assert diff <= tolerance, (
        f"Due time {parsed} not within {tolerance} of expected {expected} (diff: {diff})"
    )


# ---------------------------------------------------------------------------
# Single-turn tests (10 examples)
# ---------------------------------------------------------------------------

def _run_single_turn(message: str) -> dict:
    import uuid
    _, latency, result_messages = run_agent(message, thread_id=str(uuid.uuid4()))
    record_test_latency(latency)
    _print_trajectory("single-turn", result_messages)
    return _assert_save_reminder_called(result_messages)


def test_single_1_email_sarah_tomorrow_3pm():
    args = _run_single_turn("Remind me to email Sarah tomorrow at 3pm")
    _check_subject(args["subject"], ["email", "sarah"])
    now = datetime.now(timezone.utc)
    tomorrow_3pm = (now + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], tomorrow_3pm)


def test_single_2_water_in_15min():
    args = _run_single_turn("Remind me to drink water in 15 minutes")
    _check_subject(args["subject"], ["water"])
    expected = datetime.now(timezone.utc) + timedelta(minutes=15)
    _check_due_time(args["due_time"], expected)


def test_single_3_interview_wednesday_11am():
    args = _run_single_turn("Remind me about the interview on Wednesday at 11am")
    _check_subject(args["subject"], ["interview"])
    now = datetime.now(timezone.utc)
    days_ahead = (2 - now.weekday()) % 7
    wednesday = now + timedelta(days=days_ahead)
    wed_11am = wednesday.replace(hour=11, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], wed_11am)


def test_single_4_water_plants_today_6pm():
    args = _run_single_turn("Remind me to water plants today at 6pm")
    _check_subject(args["subject"], ["water", "plants"])
    now = datetime.now(timezone.utc)
    today_6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], today_6pm)


def test_single_5_report_july20_9am():
    args = _run_single_turn("Remind me to submit the report on July 20 at 9am")
    _check_subject(args["subject"], ["submit", "report"])
    now = datetime.now(timezone.utc)
    expected = now.replace(month=7, day=20, hour=9, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], expected)


def test_single_6_groceries_in_2hours():
    args = _run_single_turn("Remind me to buy groceries in 2 hours")
    _check_subject(args["subject"], ["groceries"])
    expected = datetime.now(timezone.utc) + timedelta(hours=2)
    _check_due_time(args["due_time"], expected)


def test_single_7_dentist_next_monday_10am():
    args = _run_single_turn("Remind me to call the dentist next Monday at 10am")
    _check_subject(args["subject"], ["dentist"])
    now = datetime.now(timezone.utc)
    days_ahead = (0 - now.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    monday = now + timedelta(days=days_ahead)
    monday_10am = monday.replace(hour=10, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], monday_10am)


def test_single_8_rent_1st_next_month():
    args = _run_single_turn("Remind me to pay rent on the 1st of next month at 9am")
    _check_subject(args["subject"], ["pay", "rent"])
    now = datetime.now(timezone.utc)
    if now.month == 12:
        expected = now.replace(year=now.year + 1, month=1, day=1, hour=9, minute=0, second=0, microsecond=0)
    else:
        expected = now.replace(month=now.month + 1, day=1, hour=9, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], expected)


def test_single_9_pickup_kids_3pm_today():
    args = _run_single_turn("Remind me to pick up the kids at 3pm today")
    _check_subject(args["subject"], ["pick", "kids"])
    now = datetime.now(timezone.utc)
    today_3pm = now.replace(hour=15, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], today_3pm)


def test_single_10_book_flights_dec15_noon():
    args = _run_single_turn("Remind me to book flights on December 15 at noon")
    _check_subject(args["subject"], ["book", "flights"])
    now = datetime.now(timezone.utc)
    expected = now.replace(month=12, day=15, hour=12, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], expected)


# ---------------------------------------------------------------------------
# Two-turn tests (10 examples)
# ---------------------------------------------------------------------------

def _run_two_turn(t1: str, t2: str) -> dict:
    import uuid
    thread_id = str(uuid.uuid4())

    # Turn 1: should NOT call save_reminder (info incomplete)
    _, lat1, result1 = run_agent(t1, thread_id=thread_id)
    _print_trajectory("two-turn (turn 1)", result1)
    _assert_save_reminder_not_called(result1)

    # Turn 2: should call save_reminder with complete info
    _, lat2, result2 = run_agent(t2, thread_id=thread_id)
    record_test_latency(lat1 + lat2)
    _print_trajectory("two-turn (turn 2)", result2)
    return _assert_save_reminder_called(result2)


def test_two_turn_1_missing_time_email_professor():
    args = _run_two_turn("Remind me to email the professor tomorrow", "at 2pm")
    _check_subject(args["subject"], ["email", "professor"])
    now = datetime.now(timezone.utc)
    tomorrow_2pm = (now + timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], tomorrow_2pm)


def test_two_turn_2_missing_time_meeting_friday():
    args = _run_two_turn("Remind me about the meeting on Friday", "at 2pm")
    _check_subject(args["subject"], ["meeting"])
    now = datetime.now(timezone.utc)
    days_ahead = (4 - now.weekday()) % 7
    friday = now + timedelta(days=days_ahead)
    friday_2pm = friday.replace(hour=14, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], friday_2pm)


def test_two_turn_3_missing_time_report_july20():
    args = _run_two_turn("Remind me to submit the report on July 20", "at 9am")
    _check_subject(args["subject"], ["submit", "report"])
    now = datetime.now(timezone.utc)
    expected = now.replace(month=7, day=20, hour=9, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], expected)


def test_two_turn_4_missing_time_water_plants():
    args = _run_two_turn("Remind me to water plants today", "at 6pm")
    _check_subject(args["subject"], ["water", "plants"])
    now = datetime.now(timezone.utc)
    today_6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], today_6pm)


def test_two_turn_5_missing_time_book_flights():
    args = _run_two_turn("Remind me to book flights on December 15", "at noon")
    _check_subject(args["subject"], ["book", "flights"])
    now = datetime.now(timezone.utc)
    expected = now.replace(month=12, day=15, hour=12, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], expected)


def test_two_turn_6_missing_subject_in_20min():
    args = _run_two_turn("Remind me in 20 minutes", "to stretch my legs")
    _check_subject(args["subject"], ["stretch"])
    expected = datetime.now(timezone.utc) + timedelta(minutes=20)
    _check_due_time(args["due_time"], expected)


def test_two_turn_7_missing_subject_saturday_7am():
    args = _run_two_turn("Remind me on Saturday at 7am", "to go for a run")
    _check_subject(args["subject"], ["run"])
    now = datetime.now(timezone.utc)
    days_ahead = (5 - now.weekday()) % 7
    saturday = now + timedelta(days=days_ahead)
    sat_7am = saturday.replace(hour=7, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], sat_7am)


def test_two_turn_8_missing_subject_in_2hours():
    args = _run_two_turn("Remind me in 2 hours", "to check the oven")
    _check_subject(args["subject"], ["check", "oven"])
    expected = datetime.now(timezone.utc) + timedelta(hours=2)
    _check_due_time(args["due_time"], expected)


def test_two_turn_9_missing_subject_friday_3pm():
    args = _run_two_turn("Remind me on Friday at 3pm", "to pick up the package")
    _check_subject(args["subject"], ["pick", "package"])
    now = datetime.now(timezone.utc)
    days_ahead = (4 - now.weekday()) % 7
    friday = now + timedelta(days=days_ahead)
    friday_3pm = friday.replace(hour=15, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], friday_3pm)


def test_two_turn_10_missing_subject_today_8pm():
    args = _run_two_turn("Remind me today at 8pm", "to take out the trash")
    _check_subject(args["subject"], ["trash"])
    now = datetime.now(timezone.utc)
    today_8pm = now.replace(hour=20, minute=0, second=0, microsecond=0)
    _check_due_time(args["due_time"], today_8pm)
