import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage

class NotesGeneratorState(TypedDict):
    # ------------------------------------------------------------------ #
    #  Conversation / tool-call history                                    #
    #  Cleared after research completes to keep downstream context lean.  #
    # ------------------------------------------------------------------ #
    messages: Annotated[List[BaseMessage], operator.add]
 
    # The topic/question passed in by the supervisor
    search_query: str
 
    # ------------------------------------------------------------------ #
    #  Research phase                                                      #
    # ------------------------------------------------------------------ #
 
    # Clean, synthesised Markdown summary produced by the researcher
    research_data: str
 
    # Set to True once the researcher is satisfied (or the iteration cap hits)
    research_complete: bool
 
    # Tracks how many researcher→tools→researcher round-trips have occurred.
    # Guards against infinite tool-call loops.
    research_iterations: int
 
    # ------------------------------------------------------------------ #
    #  Writing phase                                                       #
    # ------------------------------------------------------------------ #
 
    # Table of contents produced by the planner
    outline: List[str]
 
    # Written sections accumulated by the writer (operator.add → append)
    content_chunks: Annotated[List[str], operator.add]
 
    # Index of the section currently being written
    current_section_idx: int
 
    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #
 
    # Absolute or relative path of the generated PDF
    pdf_path: str