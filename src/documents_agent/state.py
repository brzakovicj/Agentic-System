from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class DocumentsState(TypedDict):
    """
        The state of the documents agent. 
    """
    messages: Annotated[list, add_messages] = []

    user_query: str