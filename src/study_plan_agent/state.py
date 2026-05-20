import operator
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph.message import add_messages

class StudyPlanState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.

    Fields:
        user_input   : Raw dictionary of user-provided inputs.
        requirements : Structured requirements extracted by Agent 1.
        topics       : Ordered topic breakdown produced by Agent 2.
        study_plan   : Day-wise schedule created by Agent 3.
        final_output : Polished, reviewer-approved plan from Agent 4.
        error        : Optional error message if any agent fails.
    """
    messages: Annotated[list, add_messages]

    user_input: str | None = None
    
    task_description: str | None = None

    scholar_data: Annotated[list, operator.add] = []

    agenda_data: Annotated[list, operator.add] = []