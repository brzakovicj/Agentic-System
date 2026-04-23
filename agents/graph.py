from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, add_messages, START
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from typing import List, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
import os

from agents.prompts.prompt_manager import PromptManager


class AgentState(BaseModel):
    messages: Annotated[List, add_messages]

async def build_agent_graph(tools: List[BaseTool] = []):
    prompt_manager = PromptManager()

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    if tools:
        llm = llm.bind_tools(tools)
        
        #inject tools into system prompt
        tools_json = [tool.model_dump_json(include=["name", "description"]) for tool in tools]
        tools="\n".join(tools_json)
        working_dir=os.environ.get("MCP_FILESYSTEM_DIR") if os.environ.get("MCP_FILESYSTEM_DIR") else os.getcwd()
        system_prompt = prompt_manager.get("system_prompt", tools=tools, working_dir=working_dir)

    def assistant(state: AgentState) -> AgentState:
        response = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
        state.messages.append(response)
        return 
    
    builder = StateGraph(AgentState)

    builder.add_node("Scout", assistant)
    builder.add_node(ToolNode(tools))

    builder.add_edge(START, "Scout")
    builder.add_conditional_edges(
        "Scout",
        tools_condition,
    )
    builder.add_edge("tools", "Scout")

    return builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    graph = build_agent_graph()