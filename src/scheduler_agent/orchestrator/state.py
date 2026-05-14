from typing import Annotated, TypedDict
from langgraph.graph import add_messages

from src.scheduler_agent.orchestrator.tools import TaskSchema

class OrchestratorState(TypedDict):
    """
        The state of the orchestrator agent. 
    """
    messages: Annotated[list, add_messages] = []

    task_description: str | None = None
    
    final_answer: bool = False

    plan: list[TaskSchema] = []
    
    current_task_idx: int = 0