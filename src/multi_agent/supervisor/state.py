from pydantic import BaseModel
from typing import Annotated, Optional
from langgraph.graph import add_messages

class SupervisorState(BaseModel):
    """
        The state of the supervisor agent. 
    """
    messages: Annotated[list, add_messages] = []
    researcher_answer: Optional[str] = None
    task_description: str | None = None