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
from src.agenda_agent.state import AgendaState
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

class AgendaAgent:
    def __init__(self):
        self._graph = None
        self._mcp_tools_agenda_node = None
        self._prompt_manager = PromptManager()
        self._llm_factory = LLMFactory.get_instance()

        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._mcp_dir = os.path.join(_base_dir, "mcp")
        self._agenda_config = self._get_config("config.json")
        self.mcp_client = MCPClient(self._agenda_config)

    ##############################################################################################
    # HEPLER FUNCTIONS
    ##############################################################################################

    def _get_config(self, config_name: str) -> dict:
        config_path = os.path.join(self._mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            config = json.load(f)

        scrapling_path = str(Path(sys.executable).parent / "scrapling.exe")
        
        if not Path(scrapling_path).exists():
            raise FileNotFoundError(f"scrapling.exe nije pronađen na: {scrapling_path}")

        for server in config.get("mcpServers", {}).values():
            if isinstance(server, dict) and server.get("command") == "SCRAPLING_PATH":
                server["command"] = scrapling_path

        return config

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def build_graph(self):
        self._mcp_tools_agenda_node = await self.mcp_client.get_tools()

        builder = StateGraph(AgendaState)

        # Nodes
        builder.add_node("initialize_node", self.initialize_node)
        builder.add_node("agenda_node", self.agenda_node)
        builder.add_node("agenda_tools", ToolNode(self._mcp_tools_agenda_node))

        # Edges
        builder.set_entry_point("initialize_node")

        builder.add_edge("initialize_node", "agenda_node")

        builder.add_conditional_edges(
            "agenda_node",
            self._agenda_router,
            {
                "tools": "agenda_tools",
                END: END
            }
        )

        builder.add_edge("agenda_tools", "agenda_node")

        self._graph = builder.compile(checkpointer=MemorySaver())

        return self._graph
    
    async def _agenda_router(self, state: AgendaState) -> str:
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return END
    
    ###################################################################################
    # NODE FUNCTIONS
    ###################################################################################

    async def initialize_node(self, state: AgendaState) -> AgendaState:
        # url = state.get("url")
        url = "https://imi.pmf.kg.ac.rs/pub/08b6ac285fd13a26dca006821749d364_03312026_013804/jun-septembar%20-%20informatika_2025-26_2.pdf"

        if not url:
            url = interrupt(
                {
                    "message": "Send the link from which you want me to download the schedule.",
                    "type": "text_input",
                    "placeholder": "https://example.com/raspored"
                }
            )

        ai_message = AIMessage(content = f"Successfully initialized.")

        return {
            "messages": [ai_message],
            "url": url,
        }

    async def agenda_node(self, state: AgendaState):
        # LLM
        llm = self._llm_factory.get_tool_llm(tier=ModelTier.REMOTE, tools=self._mcp_tools_agenda_node)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._mcp_tools_agenda_node
        )

        system_prompt = self._prompt_manager.get(
            "agenda_agent",
            tool_context=tools_context,
            url=state["url"] or "",
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # messages
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            print(f"Agenda agent: {exc}")
        
        return {
            "messages": [response]
        }
    
    ###############################################################
    # STREAM
    ###############################################################

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        state = AgendaState(
            messages = [ HumanMessage(content = query) ]
        )

        config = RunnableConfig(
            recursion_limit=10,
            configurable={
                "thread_id": context_id,
            }
        )

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
                                msg_content = msg.content or "[Tool returned no content]"
                            
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


#Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())