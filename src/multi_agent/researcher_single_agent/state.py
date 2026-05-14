from typing import Annotated, TypedDict

from langgraph.graph import add_messages

class ResearcherSingleAgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages] = []
    final_answer: str