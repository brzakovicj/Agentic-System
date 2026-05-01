from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
from langchain_ollama.chat_models import ChatOllama

from src.multi_agent.copywriter.state import CopyWriterState
from src.multi_agent.copywriter.tools import generate_blog_post, generate_linkedin_post, review_research_reports

load_dotenv()

# Load the copywriter system prompt and content examples
copywriter_prompt = open("src/prompts/copywriter.md", "r").read()

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

tools=[
    review_research_reports,
    generate_linkedin_post, 
    generate_blog_post
]
llm_with_tools = llm.bind_tools(tools)

async def copywriter(state: CopyWriterState):
    """The main copywriter agent."""
    system_prompt = SystemMessage(content=copywriter_prompt.format(
        current_datetime=datetime.now()
        ))
    response = llm_with_tools.invoke([system_prompt] + state.messages)
    return {"messages": [response]}

async def copywriter_router(state: CopyWriterState) -> str:
    """Route to the tools node if the copywriter makes a tool call."""
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(CopyWriterState)

builder.add_node(copywriter)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("copywriter")

builder.add_conditional_edges(
    "copywriter",
    copywriter_router,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_edge("tools", "copywriter")

# Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be inherited
graph = builder.compile()

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())