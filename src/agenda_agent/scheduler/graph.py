from datetime import datetime
import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from src.agenda_agent.scheduler.state import SchedulerState
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from langgraph.types import interrupt

load_dotenv()

class SchedulerAgent:

    def __init__(self):
        self._graph = None
        self._prompt_manager = PromptManager()
        self._factory = LLMFactory.get_instance()
        
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._mcp_dir = os.path.join(_base_dir, "mcp")

    async def initialize(self):
        _scheduler_config = self._get_config("config.json")
        
        # Cleanly replace any previous MCP client to avoid connection leaks
        if hasattr(self, "mcp_client"):
            await self.mcp_client.close()

        mcp_client = MCPClient(_scheduler_config)
        self._mcp_tools_scheduler_node = await mcp_client.get_tools()

        await self._build_graph()

        return self._graph
    
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

        # Zameni placeholder unutar mcpServers wrappera
        for server in config.get("mcpServers", {}).values():
            if isinstance(server, dict) and server.get("command") == "SCRAPLING_PATH":
                server["command"] = scrapling_path

        return config

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def _build_graph(self):
        builder = StateGraph(SchedulerState)

        # Nodes
        builder.add_node("initialize_node", self.initialize_node)
        builder.add_node("scheduler_agent", self.scheduler_agent)
        builder.add_node("scheduler_tools", ToolNode(self._mcp_tools_scheduler_node))

        # Edges
        builder.set_entry_point("initialize_node")

        builder.add_edge("initialize_node", "scheduler_agent")

        builder.add_conditional_edges(
            "scheduler_agent",
            self.scheduler_router,
            {
                "tools": "scheduler_tools",
                END: END
            }
        )

        builder.add_edge("scheduler_tools", "scheduler_agent")

        self._graph = builder.compile()

    async def scheduler_router(self, state: SchedulerState) -> str:
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return END

    ##############################################################################################
    # GRAPH NODES
    ##############################################################################################

    async def initialize_node(self, state: SchedulerState) -> SchedulerState:
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

        return {
            "url": url,
        }

    async def scheduler_agent(self, state: SchedulerState):
        # LLM
        llm = self._factory.get_tool_llm(tier=ModelTier.REMOTE, tools=self._mcp_tools_scheduler_node)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._mcp_tools_scheduler_node
        )

        system_prompt = self._prompt_manager.get(
            "scheduler_prompt",
            tool_context=tools_context,
            url=state["url"] if state["url"] else "",
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # messages
        messages = [
            SystemMessage(content=system_prompt)
        ]
    
        try:
            response: AIMessage = await llm.ainvoke(messages + state["messages"])
        except Exception as exc:
            print(f"Scheduler agent: {exc}")
        
        print("\n--- SCHEDULER AGENT ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")
        
        return {
            "messages": [response]
        }

