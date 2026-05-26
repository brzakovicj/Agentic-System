from typing import Annotated, Literal, TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

class CourseMatchSchema(TypedDict):
    matched_key: str | None

@tool
async def handoff_to_agent(
    agent_name: Literal["agenda"],
    task_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Assign a task to a agent: agenda.
    
    Args:
        agent_name: The name of the agent to handoff the task to. Valid agent names are agenda.
        task_description: The description of the task to be completed.
    """
    update = {
        "task_description": task_description,
        "messages": [ToolMessage(
            name=f"handoff_to_agent",
            content=f"Successfully handed off task to {agent_name}.",
            tool_call_id=tool_call_id,
        )],
    }

    return Command(
        goto="agenda_agent",
        update=update
    )