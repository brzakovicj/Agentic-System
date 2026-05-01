from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.multi_agent.copywriter.state import CopyWriterState

@tool
async def review_research_reports(
    state: Annotated[CopyWriterState, InjectedState],
):
    """Use this tool to review available research reports to inform your writing.
    
    Returns:
        A list of research reports.
    """
    return [report.model_dump_json() for report in state.research_reports]

@tool
async def generate_linkedin_post(
    title: str,
    content: str,
):
    """Use this tool to generate a LinkedIn post.
    
    Args:
        title: The title of the post.
        content: The content of the post in markdown format.

    Returns:
        A string indicating the location of the saved post.
    """
    filename=f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)

    return f"The LinkedIn post has been generated and saved to {filename}"

@tool
async def generate_blog_post(
    title: str,
    content: str,
):
    """Use this tool to generate a blog post.
    
    Args:
        title: The title of the post.
        content: The content of the post in markdown format.

    Returns:
        A string indicating the location of the saved post.
    """
    filename=f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)

    return f"The blog post has been generated and saved to {filename}"