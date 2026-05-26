from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class AgendaState(TypedDict):
    messages: Annotated[list, add_messages]
 
    agenda_url: str | None

    user_input: str | None