from typing import List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.multi_agent.notes_generator.state import NotesGeneratorState
from src.multi_agent.notes_generator.tools import create_pdf

import os
import json
import re

from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient

load_dotenv()

MAX_RESEARCH_ITERATIONS = 5

class NotesGeneratorAgent:
    def __init__(self):
        self.graph = None
        self.tools = None
        self.prompt_manager = PromptManager()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "mcp", "config.json")

        with open(config_path, "r") as f:
            self.mcp_config = json.load(f)

    async def initialize(self):
        """Async initialisation: connects MCP client and compiles the graph."""
        # Cleanly replace any previous MCP client to avoid connection leaks
        if hasattr(self, "mcp_client"):
            await self.mcp_client.close()  # call whatever cleanup your MCPClient exposes
        
        self.mcp_client = MCPClient(self.mcp_config)
        self.tools = await self.mcp_client.get_tools()
        self.llm_factory = LLMFactory.get_instance()

        await self._build_graph()

        return self.graph

    # ------------------------------------------------------------------ #
    #  Graph                                                             #
    # ------------------------------------------------------------------ #

    async def _build_graph(self):
        builder = StateGraph(NotesGeneratorState)

        builder.add_node("researcher", self.researcher)
        builder.add_node("planner", self.planner)
        builder.add_node("writer", self.writer)
        builder.add_node("publisher", self.publisher)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("researcher")
        builder.add_edge("planner", "writer")
        
        builder.add_conditional_edges(
            "researcher",
            self.researcher_router,
            {
                "planner": "planner",
                "tools": "tools",
                "researcher": "researcher"
            }
        )

        builder.add_conditional_edges(
            "writer",
            self.writer_router,
            {
                "writer": "writer",
                "publisher": "publisher"
            }
        )

        builder.add_edge("tools", "researcher")
        builder.add_edge("publisher", END)

        # Do NOT attach a checkpointer here — the parent supervisor graph's
        # checkpointer is inherited automatically when this is used as a subgraph.
        self.graph = builder.compile()

    # ------------------------------------------------------------------ #
    #  Nodes                                                             #
    # ------------------------------------------------------------------ #

    async def researcher(self, state: NotesGeneratorState):
        """
        Queries the vector store (and optionally the web) to collect material
        for the topic. Loops via tool calls until coverage is sufficient or
        MAX_RESEARCH_ITERATIONS is reached.
        """
        # Guard: force completion if we have been looping too long
        if state["research_iterations"] >= MAX_RESEARCH_ITERATIONS:
            # Synthesise whatever we have so far and mark done
            fallback_summary = "\n\n".join(
                msg.content
                for msg in state["messages"]
                if hasattr(msg, "content") and msg.content
            )
            return {
                "research_data": fallback_summary or "No research data collected.",
                "research_complete": True,
                # Clear messages now that research is done to keep the context
                # window lean for the planner/writer nodes.
                "messages": [],
            }

        prompt = f"""
        You are a Research Agent with access to an internal knowledge base and web resources.

        ## Objective
        Collect complete, relevant, and well-structured research material on the following topic:

        <topic>
        {state['search_query']}
        </topic>

        ## Research Strategy
        Follow this order strictly:

        1. **Internal knowledge base first** — begin by querying the internal knowledge base.
        Issue at most 3 focused queries covering different facets of the topic. If the knowledge
        base does not contain relevant material, accept that conclusion and move on — do not
        rephrase the same query indefinitely.

        2. **Web resources to fill gaps** — after the internal knowledge base is exhausted or
        conclusively unhelpful, use web resources to find and retrieve detailed information
        on the topic. Retrieve the full content of promising sources, not just surface-level
        summaries.

        ## Stopping Condition
        You have done enough research when you have gathered substantive information from
        at least 3 high-quality sources, or when further querying is clearly not producing
        new information. At that point, stop and write the report immediately.

        ## Source Discipline
        - Every claim must be traceable to a specific source.
        - Internal sources: cite file name, page number, and relevance score.
        - Web sources: cite the URL and publication date where available.
        - Discard irrelevant results — do not include them to pad coverage.

        ## Output (after ALL querying is complete)
        Produce a single dense research dump in Markdown. Do not impose thematic structure —
        the outline planner downstream will handle organization. Focus on:
        - **Completeness** — include everything relevant you found.
        - **Citations on every claim**.
        - **Gaps** — end with a short list of aspects that could not be sourced, so the
        planner knows what coverage is missing.
        """

        print("\n\n--- CURRENT MESSAGES ---\n\n")
        for m in state["messages"]:
            print(type(m), getattr(m, "content", None)[:300])

        print("\n\n----------------------\n\n")

        llm_with_tools = self.llm_factory.get_tool_llm(ModelTier.REMOTE, self.tools)
        response = await llm_with_tools.ainvoke(
            [SystemMessage(content = prompt)] + state["messages"]
        )
        
        # No tool calls → the LLM produced its final synthesis
        if not response.tool_calls:
            return {
                "messages": [],           # clear to save context window
                "research_data": response.content,
                "research_complete": True,
                "research_iterations": state["research_iterations"] + 1,
            }

        return {
            "messages": [response],
            "research_iterations": state["research_iterations"] + 1,
        }

    def researcher_router(self, state: NotesGeneratorState) -> str:
        last = state["messages"][-1] if state["messages"] else None

        # If the last message carries tool calls, dispatch them
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        
        # Research finished (either naturally or by the iteration guard)
        if state.get("research_complete"):
            return "planner"
        
        # Otherwise keep researching
        return "researcher"

    # ------------------------------------------------------------------ #
    
    async def planner(self, state: NotesGeneratorState):
        """Turns the research summary into a structured Table of Contents."""
        
        prompt = f"""
        You are an outline planner creating a Table of Contents for a student study script.

        <topic>
        {state['search_query']}
        </topic>

        <research_material>
        {state['research_data']}
        </research_material>

        Analyze the research material and produce a logical, pedagogically sound outline.
        Order sections so concepts build on each other — foundational knowledge before
        advanced applications, theory before examples.

        Rules:
        - Brief material → 3–5 sections.
        - Extensive material → 10–15 sections.
        - Do not create sections for material that wasn't found in research (check the gaps).
        - Return ONLY a JSON array of objects. No preamble, no markdown fences.

        Output format:
        [
        {{
            "title": "Section title",
            "description": "What this section covers and why it appears here in this order."
        }}
        ]
        """
        
        llm = self.llm_factory.get_remote_llm()
        response = await llm.ainvoke([SystemMessage(content = prompt)])
        outline = self._parse_json_list(response.content) 
        
        return {
            "outline": outline,
            "current_section_idx": 0
        }
    
    def _parse_json_list(self, text: str) -> List[str]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError("Could not parse outline from LLM response:\n{text}")
    
    # ------------------------------------------------------------------ #
    
    async def writer(self, state: NotesGeneratorState):
        """Writes one section of the study script per invocation."""
    
        idx = state["current_section_idx"]
        total = len(state["outline"])
        current_topic = state["outline"][idx]

        # Provide only the two most recent sections as context to avoid
        # inflating the prompt with the entire document on every iteration.
        recent_content = "\n\n".join(state["content_chunks"][-2:])
            
        prompt = f"""
        You are writing structured STUDY NOTES (not a play).
        You are writing section {idx + 1} of {total} for a student study script.
        Topic of this section: {current_topic}
        
        Previously written sections (for continuity — do NOT repeat them):
        {recent_content if recent_content else "None yet."}
        
        Research material to draw from:
        {state['research_data']}
        
        Instructions:
        - Write only what is scoped to this section's brief. The planner has already decided
        what belongs here — do not expand into other sections' territory.
        - Do not repeat content from previous sections.
        - Assume previous sections already fully explained their topics.
        - If a concept was already covered earlier, reference it briefly instead of redefining it.
        - Avoid repeating definitions, examples, or comparisons unless absolutely necessary.
        - Each section should introduce NEW information only.
        - Draw exclusively from the research material. Do not use general knowledge.
        - Every claim must include a citation: internal sources as (file name, p. X),
        web sources as (URL).

        Formatting rules:
        - USE MARKDOWN SYNTAX.
        - Use VALID Markdown tables.
        - Keep paragraphs concise (2–5 sentences max).
        - Prefer bullet points over dense prose.
        - Do not repeat concepts already covered in previous sections.
        - Use ## for section headings and ### for subsections.
        - Use Markdown code blocks for code/examples.
        - Never fake tables using spacing.
        - Always use proper Markdown table syntax.
        - Tables must contain SHORT content only.
        - Never place full paragraphs inside table cells.
        - Keep each cell under 15 words when possible.
        - Use bullet lists instead of tables for long explanations.
        """

        if idx == 0:
            prompt += "\n- Open with a brief introduction to the overall topic."
        if idx == total - 1:
            prompt += "\n- Close with a concise summary / key-takeaways section."
        
        llm = self.llm_factory.get_remote_llm()
        response = await llm.ainvoke([SystemMessage(content = prompt)])
        
        return {
            "content_chunks": [response.content], # operator.add appends
            "current_section_idx": idx + 1
        }
    
    def writer_router(self, state: NotesGeneratorState):
        """Continue writing if sections remain, otherwise publish."""

        if state["current_section_idx"] < len(state["outline"]):
            return "writer" # Go back to writing the next section
        return "publisher"  # Move to PDF creation

    # ------------------------------------------------------------------ #

    async def publisher(self, state: NotesGeneratorState):
        """Joins all written sections and renders them to a PDF."""

        full_text = "\n\n".join(state["content_chunks"])
    
        safe_query = (
            re.sub(r"[^\w\s-]", "", state["search_query"])
            .strip()
            .replace(" ", "_")
        )
        
        file_path = create_pdf(full_text)
        
        return {"pdf_path": file_path}


# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())