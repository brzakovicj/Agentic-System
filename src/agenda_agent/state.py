from typing import Annotated, NotRequired, TypedDict
from langgraph.graph import add_messages

class AgendaState(TypedDict):
    """
        The state of the agenda agent. 
    """
    messages: Annotated[list, add_messages]

    url: NotRequired[str | None]