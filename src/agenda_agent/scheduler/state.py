from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class SchedulerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] = []

    url: str

    scheduler_data: str