from typing import Annotated, Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

@tool
async def handoff_to_subagent(
    agent_name: Literal["researcher", "copywriter"],
    task_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ):
    """Assign a task to a sub-agent: researcher or copywriter.
    
    Args:
        agent_name: The name of the agent to handoff the task to. Valid agent names are researcher and copywriter.
        task_description: The description of the task to be completed.
    """
    # Construct the update schema for the Command primitive
    # We're specifying to update the task description in the state and add a tool message to the conversation to let the supervisor know the task has been handed off.
    update = {
        "task_description": task_description,
        "messages": [ToolMessage(
            name=f"handoff_to_{agent_name}",
            content=f"Successfully handed off task to {agent_name}.",
            tool_call_id=tool_call_id,
        )],
        }

    # Return the Command primitive with the update to the state and specifying the next node to go to
    return Command(
        goto=f"call_{agent_name}",
        update=update
    )