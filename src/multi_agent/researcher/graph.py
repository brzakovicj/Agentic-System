from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
from langchain_ollama.chat_models import ChatOllama
# from src.multi_agent.researcher.mcp import mcp_config

from src.multi_agent.researcher.mcp.client import MCPClient
from src.multi_agent.researcher.state import ResearcherState
from src.multi_agent.researcher.tools import extract_content_from_webpage, generate_research_report, search_web

import os
import json

load_dotenv()

class ResearcherAgent:
    def __init__(self):
        self.graph = None
        self.llm = None
        self.tools = None
        self.researcher_prompt = open("src/prompts/researcher.md", "r").read()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            base_dir,
            "mcp",
            "config.json"
        )
        with open(config_path, "r") as f:
            self.mcp_config = json.load(f)

    async def initialize(self):
        """Async init za MCP i graf."""
        # MCP tools
        self.mcp_client = MCPClient(self.mcp_config)
        mcp_tools = await self.mcp_client.get_tools()

        # Kombinuj sve alate
        self.tools = [
            search_web,
            extract_content_from_webpage,
            generate_research_report,
        ] + mcp_tools

        # LLM
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
        )

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build graph
        await self._build_graph()

        return self.graph

    async def _build_graph(self):
        """Kreira LangGraph."""
        builder = StateGraph(ResearcherState)

        builder.add_node("researcher", self.researcher)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("researcher")

        builder.add_edge("tools", "researcher")

        builder.add_conditional_edges(
            "researcher",
            self.researcher_router,
            {
                "tools": "tools",
                END: END,
            }
        )

        # Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
        self.graph = builder.compile()
    
    async def researcher(self, state: ResearcherState):
        """The main researcher agent."""
        response = self.llm_with_tools.invoke([
            SystemMessage(
                content=self.researcher_prompt.format(
                    current_datetime=datetime.now()
                )
            )
        ] + state.messages)
        
        return {"messages": [response]}

    async def researcher_router(self, state: ResearcherState) -> str:
        """Route to the tools node if the researcher makes a tool call."""
        if state.messages[-1].tool_calls:
            return "tools"
        return END
    

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())