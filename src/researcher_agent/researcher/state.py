from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class ResearcherState(TypedDict):
    query: str
    messages: Annotated[list, add_messages] = []