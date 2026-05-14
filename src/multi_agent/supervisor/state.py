import operator
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

from src.multi_agent.supervisor.tools import TaskSchema

class SupervisorState(TypedDict):
    """
        The state of the supervisor agent. 
    """
    messages: Annotated[list, add_messages] = []

    task_description: str | None = None
    
    final_answer: bool = False

    plan: list[TaskSchema] = []
    
    current_task_idx: int = 0
    
    research_data: Annotated[list, operator.add] = []