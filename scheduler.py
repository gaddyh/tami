"""Background scheduler that sends WhatsApp messages for due reminders."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from config import settings
from db import get_connection, get_placeholder, is_postgres
from dialog360 import Dialog360Client

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
STUCK_SENDING_TIMEOUT = timedelta(minutes=5)
MAX_ATTEMPTS = 3

# NOTE: This scheduler assumes a single uvicorn worker. If multiple workers
# are used, both will query and send the same reminders. The eventual fix is
# SELECT ... FOR UPDATE SKIP LOCKED (Postgres only).


def _naive_utc_now() -> datetime:
    """Current UTC time as naive datetime (for DB comparisons with SQLite)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _recover_stuck_sending(conn) -> int:
    """Reset SENDING rows that have been stuck longer than the timeout back to PENDING."""
    ph = get_placeholder()
    cutoff = (_naive_utc_now() - STUCK_SENDING_TIMEOUT).isoformat()
    now = _naive_utc_now().isoformat()

    if is_postgres():
        cursor = conn.execute(
            f"SELECT id, attempts FROM reminders "
            f"WHERE status = 'sending' AND updated_at < {ph}",
            (cutoff,),
        )
        stuck = cursor.fetchall()
    else:
        cursor = conn.execute(
            f"SELECT id, attempts FROM reminders "
            f"WHERE status = 'sending' AND updated_at < ?",
            (cutoff,),
        )
        stuck = cursor.fetchall()

    for row in stuck:
        row_id = row[0]
        attempts = row[1] + 1
        conn.execute(
            f"UPDATE reminders SET status = 'pending', attempts = {ph}, updated_at = {ph} "
            f"WHERE id = {ph}",
            (attempts, now, row_id),
        )

    if stuck:
        if not is_postgres():
            conn.commit()
        logger.warning("Recovered %d stuck SENDING reminder(s)", len(stuck))

    return len(stuck)


async def _send_due_reminders(wa: Dialog360Client) -> int:
    """Query due reminders, send them, and update status. Returns count sent."""
    now = _naive_utc_now()
    now_iso = now.isoformat()
    ph = get_placeholder()
    sent_count = 0

    conn = get_connection()
    try:
        _recover_stuck_sending(conn)

        if is_postgres():
            cursor = conn.execute(
                f"SELECT id, chat_id, subject FROM reminders "
                f"WHERE status = 'pending' AND due_time <= {ph}",
                (now_iso,),
            )
            due = cursor.fetchall()
        else:
            cursor = conn.execute(
                f"SELECT id, chat_id, subject FROM reminders "
                f"WHERE status = 'pending' AND due_time <= ?",
                (now_iso,),
            )
            due = cursor.fetchall()

        if not due:
            return 0

        for row in due:
            row_id = row[0]
            chat_id = row[1]
            subject = row[2]

            try:
                conn.execute(
                    f"UPDATE reminders SET status = 'sending', updated_at = {ph} "
                    f"WHERE id = {ph}",
                    (now_iso, row_id),
                )
                if not is_postgres():
                    conn.commit()

                fire_message = f"⏰ Reminder: {subject}"
                await wa.send_text(to=chat_id, body=fire_message)

                conn.execute(
                    f"UPDATE reminders SET status = 'sent', sent_at = {ph}, updated_at = {ph} "
                    f"WHERE id = {ph}",
                    (now_iso, now_iso, row_id),
                )
                if not is_postgres():
                    conn.commit()
                sent_count += 1
                logger.info("Sent reminder #%s to %s", row_id, chat_id)

            except Exception:
                logger.exception(
                    "Failed to send reminder #%s to %s",
                    row_id,
                    chat_id,
                )
                cursor2 = conn.execute(
                    f"SELECT attempts FROM reminders WHERE id = {ph}",
                    (row_id,),
                )
                current_attempts = cursor2.fetchone()[0]
                new_attempts = current_attempts + 1
                if new_attempts >= MAX_ATTEMPTS:
                    conn.execute(
                        f"UPDATE reminders SET status = 'failed', attempts = {ph}, updated_at = {ph} "
                        f"WHERE id = {ph}",
                        (new_attempts, now_iso, row_id),
                    )
                    logger.error(
                        "Reminder #%s marked FAILED after %d attempts",
                        row_id,
                        new_attempts,
                    )
                else:
                    conn.execute(
                        f"UPDATE reminders SET status = 'pending', attempts = {ph}, updated_at = {ph} "
                        f"WHERE id = {ph}",
                        (new_attempts, now_iso, row_id),
                    )
                if not is_postgres():
                    conn.commit()

    finally:
        conn.close()

    return sent_count


async def _reminder_loop() -> None:
    """Background loop that polls for due reminders and sends them."""
    wa = Dialog360Client(settings)
    logger.info("Reminder scheduler started (poll interval=%ds)", POLL_INTERVAL_SECONDS)

    while True:
        try:
            sent = await _send_due_reminders(wa)
            if sent:
                logger.info("Reminder scheduler: sent %d reminder(s)", sent)
        except Exception:
            logger.exception("Error in reminder scheduler cycle")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


_scheduler_task: asyncio.Task | None = None


def start_scheduler() -> None:
    """Start the reminder scheduler as a background asyncio task."""
    global _scheduler_task
    if _scheduler_task is not None:
        return
    _scheduler_task = asyncio.create_task(_reminder_loop())


async def stop_scheduler() -> None:
    """Cancel the reminder scheduler task."""
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None
