from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from langchain_core.messages import SystemMessage
from src.multi_agent.notes_generator.state import NotesGeneratorState
from src.multi_agent.notes_generator.tools import PlannerSchema, create_pdf

load_dotenv()

class NotesGeneratorAgent:
    def __init__(self):
        self.graph = None
        self.tools = None
        self.prompt_manager = PromptManager()

    async def initialize(self):
        """Async initialisation: connects MCP client and compiles the graph."""
        
        self.llm_factory = LLMFactory.get_instance()

        await self._build_graph()

        return self.graph

    # ------------------------------------------------------------------ #
    #  Graph                                                             #
    # ------------------------------------------------------------------ #

    async def _build_graph(self):
        builder = StateGraph(NotesGeneratorState)

        builder.add_node("planner", self.planner)
        builder.add_node("writer", self.writer)
        builder.add_node("publisher", self.publisher)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "writer")

        builder.add_conditional_edges(
            "writer",
            self.writer_router,
            {
                "writer": "writer",
                "publisher": "publisher"
            }
        )

        builder.add_edge("publisher", END)

        # Do NOT attach a checkpointer here — the parent supervisor graph's
        # checkpointer is inherited automatically when this is used as a subgraph.
        self.graph = builder.compile()

    # ------------------------------------------------------------------ #
    #  Nodes                                                             #
    # ------------------------------------------------------------------ #

    
    async def planner(self, state: NotesGeneratorState):
        """Turns the research summary into a structured Table of Contents."""
        
        llm = self.llm_factory.get_llm_with_structured_output(schema=PlannerSchema, tier=ModelTier.REMOTE)

        prompt = self.prompt_manager.get("notes_planner_prompt", search_query = state["search_query"], research_data = state["research_data"])

        response = await llm.ainvoke([SystemMessage(content = prompt)])

        print("\n---------- PLANNER -------------\n")
        print(response)
        print("\n-----------------------\n")

        return {
            "outline": response["outline"],
            "current_section_idx": 0
        }
    
    # ------------------------------------------------------------------ #
    
    async def writer(self, state: NotesGeneratorState):
        """Writes one section of the study script per invocation."""
    
        idx = state["current_section_idx"]
        total = len(state["outline"])
        current_section = state["outline"][idx]

        # Provide only the two most recent sections as context to avoid
        # inflating the prompt with the entire document on every iteration.
        recent_content = "\n\n".join(state["content_chunks"][-2:])
            
        prompt = self.prompt_manager.get(
            "notes_writer_prompt",
            idx = idx + 1,
            total = total,
            section_title = current_section["title"],
            section_description = current_section["description"],
            recent_content = recent_content if recent_content else "None yet.",
            research_data = state["research_data"]
        )

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
        file_path = create_pdf(full_text)
        
        return {"pdf_path": file_path}


# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())