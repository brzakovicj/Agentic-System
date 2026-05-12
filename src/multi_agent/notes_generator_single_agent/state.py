from typing import Annotated, TypedDict

from langgraph.graph import add_messages

class NotesGeneratorSingleAgentState(TypedDict):
    search_query: str
    research_data: list
    messages: Annotated[list, add_messages]
    pdf_path: str