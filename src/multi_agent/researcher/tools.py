from pydantic import BaseModel, Field

# @tool
# async def search_web(
#     query: str,
#     num_results: int = 3
#     ):
#     """Search the web and get back a list of search results including the page title, url, and a short summary of each webpage.

#     Args:
#         query: The search query.
#         num_results: The number of results to return, max is 3.

#     Returns:
#         A dictionary of the search results.
#     """
#     web_search = TavilySearch(max_results=min(num_results, 3), topic="general")
#     search_results = web_search.invoke(input={"query": query})
    
#     processed_results = {
#         "query": query,
#         "results": []
#     }

#     # Light processing of the search results to return a subset of the data
#     for result in search_results["results"]:
#         processed_results["results"].append({
#             "title": result["title"],
#             "url": result["url"],
#             "content_preview": result["content"]
#         })

#     return processed_results


# @tool
# async def extract_content_from_webpage(urls: List[str]):
#     """Extract the content from a webpage.

#     Args:
#         url: The url of the webpage to extract content from.

#     Returns:
#         A list of dictionaries containing the extracted content from each webpage.
#     """
#     web_extract = TavilyExtract()
#     results = web_extract.invoke(input={"urls": urls})["results"]
#     return results

# class ResearchReport(BaseModel):
#     topic: str
#     report: str

# @tool
# async def generate_research_report(
#     topic: str,
#     report: str,
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     ):
#     """Generate a research report on a specific topic.

#     Args:
#         topic: The topic to research.
#         report: The research report.
    
#     Returns:
#         The research report.
#     """
#     research_report = ResearchReport.model_validate({
#         "topic": topic,
#         "report": report
#         })

#     # We use the Command primitive to update the state with the research report and add a tool message to the conversation with the generated report.
#     return Command(update={
#         "research_reports": [research_report],
#         "messages": [ToolMessage(
#             name="generate_research_report",
#             content=research_report.model_dump_json(),
#             tool_call_id=tool_call_id,
#             )],
#         })

class EvaluatorDecision(BaseModel):
    need_web_search: bool = Field(
        description="True if web search is needed, False if retrieved documents are sufficient"
    )
    reason: str = Field(
        description="Brief explanation of the decision"
    )