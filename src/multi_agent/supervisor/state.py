import operator
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph import add_messages

from src.multi_agent.supervisor.tools import TaskSchema

class SupervisorState(BaseModel):
    """
        The state of the supervisor agent. 
    """
    messages: Annotated[list, add_messages] = []

    task_description: str | None = None
    
    final_answer: bool = False

    plan: list[TaskSchema] = []
    
    current_task_idx: int = 0
    
    research_data: str = ""