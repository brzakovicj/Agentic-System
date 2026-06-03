import json
import os
from pathlib import Path
import sys
from typing import Any, AsyncIterable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from dotenv import load_dotenv
from langgraph.types import RunnableConfig, interrupt
from src.documents_agent.state import DocumentsState
from src.utils.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

class DocumentsAgent:
    def __init__(self):
        self.graph = None
        self.mcp_tools = None
        self.prompt_manager = PromptManager()
        self.llm_factory = LLMFactory.get_instance()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.mcp_dir = os.path.join(base_dir, "mcp")
        self.mcp_config = self.get_config("config.json")
        self.mcp_client = MCPClient(self.mcp_config)

    ##############################################################################################
    # HEPLER FUNCTIONS
    ##############################################################################################

    def get_config(self, config_name: str) -> dict:
        """Utility function to load MCP tool configuration from a JSON file."""
        config_path = os.path.join(self.mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            return json.load(f)

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def build_graph(self):
        self.mcp_tools = await self.mcp_client.get_tools()

        builder = StateGraph(DocumentsState)

        # Nodes
        builder.add_node("documents_node", self.documents_node)
        builder.add_node("tools", ToolNode(self.mcp_tools))

        # Edges
        builder.set_entry_point("documents_node")

        builder.add_conditional_edges(
            "documents_node",
            self.documents_router,
            {
                "tools": "tools",
                END: END
            }
        )

        builder.add_edge("tools", "documents_node")

        self._graph = builder.compile(checkpointer = MemorySaver())

        return self._graph
    
    async def documents_router(self, state: DocumentsState) -> str:
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return END
    
    ###################################################################################
    # NODE FUNCTIONS
    ###################################################################################

    async def documents_node(self, state: DocumentsState):
        # LLM
        llm = self.llm_factory.get_tool_llm(tier = ModelTier.REMOTE, tools = self.mcp_tools)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self.mcp_tools
        )

        system_prompt = self.prompt_manager.get(
            "documents_agent",
            user_query = state["user_query"],
            tool_context = tools_context
        )

        try:
            response: AIMessage = await llm.ainvoke([SystemMessage(content = system_prompt)] + state["messages"])
        except Exception as exc:
            print(f"Documents agent: {exc}")
        
        print("\n--- DOCUMENTS NODE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")
        
        return {
            "messages": [response]
        }
    
    ###############################################################
    # STREAM
    ###############################################################

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        state = DocumentsState(
            messages = [ HumanMessage(content = query) ],
            user_query = query
        )

        config = RunnableConfig(configurable={
            "thread_id": context_id,
            "recursion_limit": 50,
        })

        last_ai_content = ""
        completed_normally = False

        try:
            async for item in self._graph.astream(
                input = state,
                stream_mode="updates",
                config = config
            ):
                # updates je dict: {node_name: {"messages": [...]}}
                for node_name, state_update in item.items():
                    messages = state_update.get("messages", [])

                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"  TOOL CALL [{node_name}]: {tc['name']}")
                                    print(f"  {tc['args']}")
                                print()

                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': 'Calling tools...',
                                }

                            elif msg.content:
                                last_ai_content = msg.content.strip()
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': last_ai_content,
                                }

                        elif isinstance(msg, ToolMessage):
                            print(
                                f"[Tool result: {msg.name}]"
                            )
                            print()
                            
                            if (isinstance(msg.content, list)):
                                tool_parts = msg.content
                                msg_content = "\n\n".join(
                                    m.content for m in tool_parts
                                    if hasattr(m, "content") and m.content
                                )
                            else:
                                msg_content = msg.content
                            
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': 'Tool responded with results: \n' + msg_content,
                            }

            completed_normally = True

        except Exception as exc:
            print(f"Graph execution failed: {exc}")

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(exc)}",
            }

        finally:
            if completed_normally:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_ai_content,
                }

    async def close(self):
        """Call this when the agent is done."""
        await self.mcp_client.close()

#Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())