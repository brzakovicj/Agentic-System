import operator
from typing import Annotated, NotRequired, TypedDict
from langgraph.graph.message import add_messages

class StudyPlanState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.
    """
    messages: Annotated[list, add_messages]

    user_input: NotRequired[str | None]
    
    task_description: NotRequired[str | None]

    selected_agent: NotRequired[str | None]

    scholar_data: Annotated[list, operator.add]

    agenda_data: Annotated[list, operator.add]