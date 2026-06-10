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

    agenda_data: Annotated[list, operator.add]

    course_context: NotRequired[dict | None]

    agenda_context_id: NotRequired[str | None]

    agenda_message_id: NotRequired[str | None]

    agenda_status: NotRequired[str | None]

    agenda_prompt_text: NotRequired[str | None]

    syllabus_loaded: NotRequired[bool]

    syllabus_url: NotRequired[str | None]