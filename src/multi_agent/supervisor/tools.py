from typing import Annotated, Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

@tool
async def handoff_to_subagent(
    agent_name: Literal["researcher"],
    task_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ):
    """Assign a task to a sub-agent: researcher.
    
    Args:
        agent_name: The name of the agent to handoff the task to. Valid agent names are researcher.
        task_description: The description of the task to be completed.
    """
    update = {
        "task_description": task_description,
        "messages": [ToolMessage(
            name=f"handoff_to_subagent",
            content=f"Successfully handed off task to {agent_name}.",
            tool_call_id=tool_call_id,
        )],
        }

    return Command(
        goto=f"call_{agent_name}",
        update=update
    )