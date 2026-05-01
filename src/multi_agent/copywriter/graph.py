import json
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
from langchain_ollama.chat_models import ChatOllama

from src.multi_agent.copywriter.state import CopyWriterState
from src.multi_agent.copywriter.tools import generate_blog_post, generate_linkedin_post, review_research_reports
from src.multi_agent.copywriter.mcp.client import MCPClient
from src.prompts.prompt_manager import PromptManager

load_dotenv()

class CopywriterAgent:
    def __init__(self):
        self.graph = None
        self.llm = None
        self.tools = None
        self.prompt_manager = PromptManager()
        self.copywriter_prompt = None
        
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
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in mcp_tools
        )

        self.copywriter_prompt = self.prompt_manager.get("copywriter", tools = tools_context, current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Kombinuj sve alate
        self.tools=[
            review_research_reports,
            generate_linkedin_post, 
            generate_blog_post
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
        builder = StateGraph(CopyWriterState)

        builder.add_node(self.copywriter)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("copywriter")

        builder.add_conditional_edges(
            "copywriter",
            self.copywriter_router,
            {
                "tools": "tools",
                END: END,
            }
        )
        builder.add_edge("tools", "copywriter")

        # Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
        self.graph = builder.compile()
    
    async def copywriter(self, state: CopyWriterState):
        """The main copywriter agent."""
        system_prompt = SystemMessage(content = self.copywriter_prompt)
        response = self.llm_with_tools.invoke([system_prompt] + state.messages)
        return {"messages": [response]}

    async def copywriter_router(self, state: CopyWriterState) -> str:
        """Route to the tools node if the copywriter makes a tool call."""
        if state.messages[-1].tool_calls:
            return "tools"
        return END
    

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())