from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
from langchain_ollama.chat_models import ChatOllama

from src.multi_agent.researcher.state import ResearcherState
from src.multi_agent.researcher.tools import extract_content_from_webpage, generate_research_report, search_web

load_dotenv()

# Load the researcher system prompt
researcher_prompt = open("src/prompts/researcher.md", "r").read()

tools = [
    search_web, 
    extract_content_from_webpage,
    generate_research_report,
    ]

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)

async def researcher(state: ResearcherState):
    """The main researcher agent."""
    response = llm_with_tools.invoke([
        SystemMessage(content=researcher_prompt.format(current_datetime=datetime.now()))
        ] + state.messages)
    return {"messages": [response]}

async def researcher_router(state: ResearcherState) -> str:
    """Route to the tools node if the researcher makes a tool call."""
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(ResearcherState)

builder.add_node(researcher)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("researcher")
builder.add_edge("tools", "researcher")
builder.add_conditional_edges(
    "researcher",
    researcher_router,
    {
        "tools": "tools",
        END: END,
    }
)

# Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
graph = builder.compile()

# graph = builder.compile(checkpointer=MemorySaver())

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())