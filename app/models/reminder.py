from datetime import datetime

from pydantic import BaseModel, Field


class Reminder(BaseModel):
    chat_id: str = Field(..., description="Chat/thread identifier this reminder belongs to")
    subject: str = Field(..., description="What the reminder is about")
    due_time: datetime = Field(..., description="When the reminder is due, UTC naive")
    status: str = Field("pending", description="Reminder status: pending, sending, sent, failed")
