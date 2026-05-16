import operator
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

from src.scholar_agent.scholar.tools import TaskSchema

class ScholarState(TypedDict):
    """
        The state of the scholar agent. 
    """
    messages: Annotated[list, add_messages] = []

    task_description: str | None = None
    
    final_answer: bool = False

    plan: list[TaskSchema] = []
    
    current_task_idx: int = 0
    
    research_data: Annotated[list, operator.add] = []