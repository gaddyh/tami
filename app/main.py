import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import langsmith as ls
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.agent import run_agent
from app.config import settings
from app.scheduler.reminder_scheduler import start_scheduler, stop_scheduler
from app.services.dialog360 import Dialog360Client, iter_incoming_messages
from app.services.transcribe import (
    Transcriber,
    handle_360dialog_audio_message,
)
from app.services.transcription_factory import get_transcriber

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO)
)

for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

wa_client: Dialog360Client | None = None
transcriber: Transcriber | None = None

_processed_msg_ids: set[str] = set()
_PROCESSED_MSG_ID_MAX = 5000


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wa_client, transcriber

    wa_client = Dialog360Client(settings)

    transcriber = get_transcriber(settings)

    start_scheduler()

    yield

    await stop_scheduler()

    if transcriber is not None:
        await transcriber.close()


app = FastAPI(lifespan=lifespan)


def sanitize_process_message_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Prevent the sender phone number and raw provider payload from being traced."""
    msg = inputs.get("msg", {})

    return {
        "msg": {
            "id": msg.get("id"),
            "type": msg.get("type"),
            "has_text": bool(msg.get("text")),
            "has_media_id": bool(msg.get("media_id")),
            "mime_type": msg.get("mime_type"),
        }
    }


@ls.traceable(
    name="process_whatsapp_message",
    run_type="chain",
    tags=["tami", "whatsapp", "reminder-agent"],
    metadata={
        "channel": "whatsapp",
        "provider": "360dialog",
        "agent_type": "reminder",
        "app_version": "v1",
    },
    process_inputs=sanitize_process_message_inputs,
)
async def process_message(msg: dict[str, Any]) -> None:
    sender = msg["from"]
    msg_id = msg["id"]
    msg_type = msg["type"]

    run = ls.get_current_run_tree()

    if run is not None:
        run.metadata.update(
            {
                "environment": getattr(
                    settings,
                    "environment",
                    "development",
                ),
                "message_type": msg_type,
                "provider_message_id": msg_id,
            }
        )

    if msg_id in _processed_msg_ids:
        logger.info(
            "Skipping duplicate message id=%s from=%s",
            msg_id,
            sender,
        )

        if run is not None:
            run.metadata["processing_status"] = "duplicate"

        return

    _processed_msg_ids.add(msg_id)

    if len(_processed_msg_ids) > _PROCESSED_MSG_ID_MAX:
        _processed_msg_ids.clear()
        _processed_msg_ids.add(msg_id)

    try:
        if wa_client is not None:
            try:
                await wa_client.send_typing_indicator(msg_id)
            except Exception:
                logger.warning(
                    "Failed to send typing indicator for msg=%s",
                    msg_id,
                )

        if msg_type == "text":
            user_text = msg.get("text", "")

            if not user_text.strip():
                if run is not None:
                    run.metadata["processing_status"] = "empty_text"

                return

        elif msg_type == "audio":
            if transcriber is None:
                logger.error(
                    "No transcriber available — "
                    "skipping audio message from=%s",
                    sender,
                )

                if run is not None:
                    run.metadata["processing_status"] = (
                        "transcriber_unavailable"
                    )

                return

            media_id = msg.get("media_id", "")
            mime_type = msg.get("mime_type", "")

            if not media_id:
                logger.error(
                    "Audio message missing media_id from=%s",
                    sender,
                )

                if run is not None:
                    run.metadata["processing_status"] = (
                        "missing_media_id"
                    )

                return

            user_text = await handle_360dialog_audio_message(
                wa=wa_client,
                transcriber=transcriber,
                media_id=media_id,
                mime_type=mime_type,
            )

            if not user_text.strip():
                logger.info(
                    "Empty transcript from=%s — skipping",
                    sender,
                )

                if run is not None:
                    run.metadata["processing_status"] = (
                        "empty_transcript"
                    )

                return

        else:
            if run is not None:
                run.metadata["processing_status"] = (
                    "unsupported_message_type"
                )

            return

        if run is not None:
            run.metadata["input_source"] = msg_type

        response_text, latency, _ = await asyncio.to_thread(
            run_agent,
            user_text,
            thread_id=sender,
        )

        logger.info(
            "Agent responded to=%s latency=%.2fs",
            sender,
            latency,
        )

        if run is not None:
            run.metadata.update(
                {
                    "agent_latency_seconds": latency,
                    "agent_response_created": True,
                }
            )

        if wa_client is not None:
            await wa_client.send_text(
                to=sender,
                body=response_text,
            )

        if run is not None:
            run.metadata["processing_status"] = "completed"

    except Exception as exc:
        logger.exception(
            "Failed to process message from=%s type=%s",
            sender,
            msg_type,
        )

        if run is not None:
            run.metadata.update(
                {
                    "processing_status": "failed",
                    "error_type": type(exc).__name__,
                }
            )

        # We currently preserve the original behavior and swallow the error.
        # Re-raise here later if you want LangSmith to mark the root span red.
        return


@app.post("/webhook/360dialog")
async def webhook(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        logger.error("Failed to parse webhook payload as JSON")

        return JSONResponse(
            status_code=200,
            content={"status": "ok"},
        )

    messages = list(iter_incoming_messages(payload))

    if not messages:
        return JSONResponse(
            status_code=200,
            content={"status": "ok"},
        )

    for msg in messages:
        asyncio.create_task(process_message(msg))

    return JSONResponse(
        status_code=200,
        content={"status": "ok"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}