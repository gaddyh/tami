import os
import re

_DEFAULT_DB_URL = "sqlite:///reminders.db"


def get_db_url() -> str | None:
    return os.getenv("DATABASE_URL")


def is_postgres(db_url: str | None = None) -> bool:
    url = db_url or get_db_url() or ""
    return url.startswith("postgresql://") or url.startswith("postgres://")


def is_sqlite(db_url: str | None = None) -> bool:
    url = db_url or get_db_url() or ""
    return url.startswith("sqlite://")


def get_connection():
    url = get_db_url() or _DEFAULT_DB_URL
    if is_postgres(url):
        import psycopg

        conn = psycopg.connect(url, autocommit=True)
        return conn
    else:
        import sqlite3

        path = re.sub(r"^sqlite://", "", url)
        conn = sqlite3.connect(path)
        return conn


def get_placeholder() -> str:
    return "%s" if is_postgres() else "?"


def init_reminders_table(conn) -> None:
    if is_postgres():
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id SERIAL PRIMARY KEY,
                chat_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                due_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                sent_at TIMESTAMP WITHOUT TIME ZONE,
                updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_chat_id ON reminders(chat_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_due_time ON reminders(due_time)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)"
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                due_time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                sent_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_chat_id ON reminders(chat_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_due_time ON reminders(due_time)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)"
        )
        conn.commit()


def get_checkpointer(db_url: str | None = None):
    url = db_url or get_db_url()
    if not url:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if is_postgres(url):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        import psycopg

        conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return checkpointer
    elif url.startswith("sqlite://"):
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        path = re.sub(r"^sqlite://", "", url)
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn)
    else:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
