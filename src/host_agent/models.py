from typing import Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    context_id: str | None = None

class ChatResponse(BaseModel):
    type: Optional[str] = None
    content: Optional[str] = None