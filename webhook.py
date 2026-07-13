import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import run_agent
from config import settings
from dialog360 import Dialog360Client, iter_incoming_messages
from scheduler import start_scheduler, stop_scheduler
from transcribe import OpenAITranscriber, Transcriber, handle_360dialog_audio_message

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

wa_client: Dialog360Client | None = None
transcriber: Transcriber | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wa_client, transcriber

    wa_client = Dialog360Client(settings)

    if settings.transcription_provider == "openai":
        transcriber = OpenAITranscriber(
            api_key=settings.openai_api_key,
            model=settings.openai_transcribe_model,
        )
    else:
        logger.warning(
            "Transcription provider '%s' not implemented — audio messages will fail",
            settings.transcription_provider,
        )

    start_scheduler()

    yield

    await stop_scheduler()
    if transcriber is not None:
        await transcriber.close()


app = FastAPI(lifespan=lifespan)


async def process_message(msg: dict[str, Any]) -> None:
    sender = msg["from"]
    msg_id = msg["id"]
    msg_type = msg["type"]

    try:
        if wa_client:
            try:
                await wa_client.send_typing_indicator(msg_id)
            except Exception:
                logger.warning("Failed to send typing indicator for msg=%s", msg_id)

        if msg_type == "text":
            user_text = msg.get("text", "")
            if not user_text.strip():
                return

        elif msg_type == "audio":
            if transcriber is None:
                logger.error("No transcriber available — skipping audio message from=%s", sender)
                return

            media_id = msg.get("media_id", "")
            mime_type = msg.get("mime_type", "")
            if not media_id:
                logger.error("Audio message missing media_id from=%s", sender)
                return

            user_text = await handle_360dialog_audio_message(
                wa=wa_client,
                transcriber=transcriber,
                media_id=media_id,
                mime_type=mime_type,
            )
            if not user_text.strip():
                logger.info("Empty transcript from=%s — skipping", sender)
                return
        else:
            return

        response_text, latency, _ = await asyncio.to_thread(
            run_agent, user_text, thread_id=sender
        )
        logger.info("Agent responded to=%s latency=%.2fs", sender, latency)

        if wa_client:
            await wa_client.send_text(to=sender, body=response_text)

    except Exception:
        logger.exception("Failed to process message from=%s type=%s", sender, msg_type)


@app.post("/webhook/360dialog")
async def webhook(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        logger.error("Failed to parse webhook payload as JSON")
        return JSONResponse(status_code=200, content={"status": "ok"})

    messages = list(iter_incoming_messages(payload))
    if not messages:
        return JSONResponse(status_code=200, content={"status": "ok"})

    for msg in messages:
        asyncio.create_task(process_message(msg))

    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
