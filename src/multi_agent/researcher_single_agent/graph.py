import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from src.multi_agent.researcher_single_agent.state import ResearcherSingleAgentState
from src.utils.mcp_client import MCPClient
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier

load_dotenv()

class ResearcherSingleAgent:
    def __init__(self):
        self._graph = None
        self._prompt_manager = PromptManager()
        self.factory = LLMFactory.get_instance()
        
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._mcp_dir = os.path.join(_base_dir, "mcp")

    async def initialize(self):
        mcp_config = self._get_config("config.json")
        
        # Cleanly replace any previous MCP client to avoid connection leaks
        if hasattr(self, "mcp_client"):
            await self.mcp_client.close()

        mcp_client = MCPClient(mcp_config)
        self._mcp_tools = await mcp_client.get_tools()

        await self._build_graph()

        return self._graph

    ##############################################################################################
    # HEPLER FUNCTIONS
    ##############################################################################################

    def _get_config(self, config_name: str) -> dict:
        """Utility function to load MCP tool configuration from a JSON file."""
        config_path = os.path.join(self._mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            return json.load(f)

    def _stringify_content(self, content) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, (dict, list)):
            try:
                return json.dumps(
                    content,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:
                return str(content)

        return str(content)

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def _build_graph(self):
        """Build the state graph for the ResearcherAgent."""
        builder = StateGraph(ResearcherSingleAgentState)

        builder.add_node("researcher", self.researcher_node)
        builder.add_node("tools", ToolNode(self._mcp_tools))

        builder.set_entry_point("researcher")

        builder.add_conditional_edges(
            "researcher",
            self.router,
            {
                "tools": "tools",
                "end": END,
            }
        )

        builder.add_edge("tools", "researcher")

        # Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
        self._graph = builder.compile()

    async def router(self, state: ResearcherSingleAgentState) -> str:
        last_msg = state["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return "end"

    ##############################################################################################
    # GRAPH NODES
    ##############################################################################################

    async def researcher_node(self, state: ResearcherSingleAgentState):
        """
            DOES EVERYTHING
        """
        # LLM
        llm = self.factory.get_tool_llm(tier=ModelTier.REMOTE, tools=self._mcp_tools)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._mcp_tools
        )

        prompt = self._prompt_manager.get(
            "researcher_single_agent_prompt", 
            query = state["query"], 
            tools = tools_context, 
            current_datetime = datetime.now().strftime("%Y-%m-%d")
        )
        
        # Build messages
        messages = [
            SystemMessage(content=prompt)
        ] + state["messages"]

        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            return {
                "error": f"LLM call failed: {e}"
            }
        
        print("\n--- RESEARCHER STATE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")

        result = {
            "messages": [response]
        }

        if not getattr(response, "tool_calls", None):
            result["final_answer"] = response.content

        return result

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())