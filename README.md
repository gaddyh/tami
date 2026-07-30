# Tami — WhatsApp Reminder Assistant

Tami is a WhatsApp-based reminder assistant powered by a LangGraph ReAct agent. Users send natural language messages (text or voice) in Hebrew or English, and the agent extracts the subject and due time, saves the reminder to a database, and fires a WhatsApp notification when the reminder is due.

## Architecture

```
WhatsApp User
    │
    ▼
360dialog Webhook ──► FastAPI (webhook.py)
    │                       │
    │                       ├─ Text message → run_agent()
    │                       └─ Audio message → download → transcribe → run_agent()
    │                                              │
    ▼                                              ▼
Dialog360Client (dialog360.py)              LangGraph ReAct Agent (agent.py)
    │                                              │
    │                                              ├─ save_reminder tool (tools.py)
    │                                              │     └─ converts tenant TZ → UTC
    │                                              │     └─ inserts into reminders table
    │                                              └─ Checkpointer (Memory/SQLite/Postgres)
    │
    ▼
Background Scheduler (scheduler.py)
    │  polls every 30s for due reminders
    │  sends ⏰ message via 360dialog
    │  retries up to 3 times, recovers stuck sends
```

## Project Structure

| File | Purpose |
|---|---|
| `webhook.py` | FastAPI app with `/webhook/360dialog` POST and `/health` GET endpoints. Starts scheduler in lifespan. Deduplicates messages by ID. |
| `agent.py` | LangGraph ReAct agent with `save_reminder` tool. Rotates checkpointer thread ID after each successful save to start a fresh conversation. |
| `tools.py` | `save_reminder` LangChain tool. Converts naive LLM datetime from tenant timezone to UTC before storing. |
| `scheduler.py` | Background asyncio task that polls every 30s for due `pending` reminders, sends them via 360dialog, and manages retry/stuck-sending logic. |
| `dialog360.py` | 360dialog WhatsApp API client — send text, typing indicators, download media. Includes `iter_incoming_messages` payload parser. |
| `transcribe.py` | Audio transcription via OpenAI API (wrapped with LangSmith tracing). Handles format conversion with ffmpeg. |
| `config.py` | `Settings` dataclass loaded from environment variables. |
| `db.py` | Database connection (SQLite/Postgres), table initialization, LangGraph checkpointer factory. |
| `models.py` | Pydantic `Reminder` model. |
| `main.py` | CLI entry point for local testing without WhatsApp. |
| `test_agent_he.py` | 20 Hebrew E2E tests (single-turn + two-turn) validating subject extraction and due time accuracy. |
| `test_agent.py` | English E2E tests. |

## Setup

### Prerequisites

- Python 3.13+
- `ffmpeg` installed (for audio transcription)
- A 360dialog account with WhatsApp Business API access
- An OpenAI API key

### Installation

```bash
git clone https://github.com/gaddyh/tami.git
cd tami
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

**Required:**
- `D360_API_KEY` — your 360dialog API key
- `OPENAI_API_KEY` — your OpenAI API key

**Key settings:**
- `DATABASE_URL` — `sqlite:///reminders.db` for dev, `postgresql://...` for production
- `TENANT_TIMEZONE` — timezone for interpreting user times (default: `Asia/Jerusalem`)
- `TRANSCRIPTION_PROVIDER` — `openai` (default) or `modal`
- `LOG_LEVEL` — `INFO` (default), `DEBUG`, `WARNING`, etc.

See `.env.example` for the full list.

## Running

### Webhook server (production)

```bash
uvicorn webhook:app --host 0.0.0.0 --port $PORT --workers 1
```

> **Important:** Use `--workers 1` only. The scheduler runs as a background task inside the webhook process — multiple workers will cause duplicate reminder delivery.

### CLI mode (local testing)

```bash
python main.py
```

### Running tests

```bash
python -m pytest test_agent_he.py -v
python -m pytest test_agent.py -v
```

Tests require `OPENAI_API_KEY` to be set. They make real LLM calls and validate subject extraction + due time accuracy with tolerance.

## Webhook Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/webhook/360dialog` | Receives 360dialog webhook payloads (text + audio messages). Returns `200 OK` immediately; processing is async. |
| `GET` | `/health` | Health check. Returns `{"status": "ok"}`. |

## How It Works

1. **User sends a WhatsApp message** (text or voice note) → 360dialog delivers it to `/webhook/360dialog`
2. **Webhook responds `200 OK` immediately** and processes the message in a background task (avoids 360dialog retries)
3. **Text messages** go directly to the agent. **Audio messages** are downloaded, converted if needed, transcribed via OpenAI, then sent to the agent.
4. **The agent** (LangGraph ReAct with `gpt-5.4-mini`) extracts the subject and due time from the natural language message. If either is missing, it asks a follow-up question. When both are present, it calls `save_reminder`.
5. **`save_reminder`** interprets the LLM's naive datetime as the tenant timezone (e.g. `Asia/Jerusalem`), converts to UTC, and inserts into the `reminders` table with `status='pending'`.
6. **After a successful save**, the checkpointer thread ID is rotated so the next message starts a fresh conversation.
7. **The scheduler** polls every 30 seconds for `pending` reminders where `due_time <= now`. It marks them `sending`, sends `⏰ Reminder: {subject}` via 360dialog, then marks them `sent`. Failed sends retry up to 3 times before being marked `failed`. Stuck `sending` rows older than 5 minutes are recovered.

## Database Schema

```sql
CREATE TABLE reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- SERIAL for Postgres
    chat_id     TEXT NOT NULL,                       -- sender phone number
    subject     TEXT NOT NULL,                       -- reminder subject
    due_time    TEXT NOT NULL,                       -- UTC naive datetime
    created_at  TEXT NOT NULL,                       -- UTC naive datetime
    status      TEXT NOT NULL DEFAULT 'pending',     -- pending | sending | sent | failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    sent_at     TEXT,                                -- UTC naive, nullable
    updated_at  TEXT NOT NULL                        -- UTC naive datetime
);
```

## Deployment

**Build command:**
```bash
pip install -r requirements.txt
```

**Start command:**
```bash
uvicorn webhook:app --host 0.0.0.0 --port $PORT --workers 1
```

**Requirements:**
- `ffmpeg` must be available on the deployment image
- Use PostgreSQL (`DATABASE_URL=postgresql://...`) — SQLite won't survive ephemeral filesystems
- Set the webhook URL in your 360dialog dashboard to `https://your-domain/webhook/360dialog`

## Tech Stack

- **FastAPI** — webhook server
- **LangGraph** — ReAct agent with checkpointing
- **LangChain** — tool integration
- **OpenAI** — LLM (`gpt-5.4-mini`) + audio transcription (`gpt-4o-transcribe`)
- **360dialog** — WhatsApp Business API provider
- **httpx** — async HTTP client for 360dialog API
- **SQLite / PostgreSQL** — reminder storage + LangGraph checkpointer
- **LangSmith** — optional LLM tracing/observability
