import operator
from typing import Annotated, NotRequired, TypedDict
from langgraph.graph import add_messages

from src.scholar_agent.tools import TaskSchema

class ScholarState(TypedDict):
    """
        The state of the scholar agent. 
    """
    messages: Annotated[list, add_messages]

    task_description: NotRequired[str | None]
    
    final_answer: NotRequired[bool]

    plan: NotRequired[list[TaskSchema]]
    
    current_task_idx: NotRequired[int]
    
    research_data: Annotated[list, operator.add]

    researcher_messages: Annotated[list, add_messages]

    notes_text: NotRequired[str | None]