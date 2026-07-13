from datetime import datetime, timezone

from langchain_core.tools import tool

from db import get_connection, get_placeholder, init_reminders_table
from models import Reminder

_current_chat_id: str | None = None


def set_chat_id(chat_id: str) -> None:
    global _current_chat_id
    _current_chat_id = chat_id


@tool
def save_reminder(
    subject: str,
    due_time: str,
) -> str:
    """Save a reminder with a subject and due time to the database.

    Args:
        subject: What the reminder is about (e.g. "call mom").
        due_time: When the reminder is due, as an ISO 8601 datetime string
                  (e.g. "2025-07-15T17:00:00").
    """
    if _current_chat_id is None:
        raise RuntimeError("chat_id not set — call set_chat_id before invoking agent")

    reminder = Reminder(
        chat_id=_current_chat_id,
        subject=subject,
        due_time=datetime.fromisoformat(due_time),
    )

    conn = get_connection()
    try:
        init_reminders_table(conn)
        ph = get_placeholder()
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        from db import is_postgres
        if is_postgres():
            cursor = conn.execute(
                f"INSERT INTO reminders (chat_id, subject, due_time, created_at) "
                f"VALUES ({ph}, {ph}, {ph}, {ph}) RETURNING id",
                (reminder.chat_id, reminder.subject, reminder.due_time.replace(tzinfo=None).isoformat(), now),
            )
            reminder_id = cursor.fetchone()[0]
        else:
            cursor = conn.execute(
                f"INSERT INTO reminders (chat_id, subject, due_time, created_at) "
                f"VALUES ({ph}, {ph}, {ph}, {ph})",
                (reminder.chat_id, reminder.subject, reminder.due_time.replace(tzinfo=None).isoformat(), now),
            )
            conn.commit()
            reminder_id = cursor.lastrowid
    finally:
        conn.close()

    return f"Reminder #{reminder_id} saved: '{reminder.subject}' due at {reminder.due_time.isoformat()}"
