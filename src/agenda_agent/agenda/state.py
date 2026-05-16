from typing import Annotated, TypedDict
from langgraph.graph import add_messages

from src.agenda_agent.agenda.tools import TaskSchema

class AgendaState(TypedDict):
    """
        The state of the agenda agent. 
    """
    messages: Annotated[list, add_messages] = []

    task_description: str | None = None
    
    final_answer: bool = False

    plan: list[TaskSchema] = []
    
    current_task_idx: int = 0