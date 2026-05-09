import operator

from pydantic import BaseModel
from typing import Annotated, Optional
from langgraph.graph import add_messages

class SupervisorState(BaseModel):
    """
        The state of the supervisor agent. 
    """
    messages: Annotated[list, add_messages] = []
    task_description: str | None = None
    final_answer: bool = False