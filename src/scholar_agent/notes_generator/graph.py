from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.scholar_agent.notes_generator.state import NotesGeneratorState
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory
from langchain_core.messages import SystemMessage
from src.scholar_agent.notes_generator.tools import create_pdf

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

        builder.add_node("agent", self.agent)
        builder.add_node("publisher", self.publisher)

        builder.set_entry_point("agent")
        builder.add_edge("agent", "publisher")
        builder.add_edge("publisher", END)

        # Do NOT attach a checkpointer here — the parent supervisor graph's
        # checkpointer is inherited automatically when this is used as a subgraph.
        self.graph = builder.compile()

    # ------------------------------------------------------------------ #
    #  Nodes                                                             #
    # ------------------------------------------------------------------ #

    async def agent(self, state: NotesGeneratorState):

        llm = self.llm_factory.get_remote_llm()

        research_data_str = "\n".join(state["research_data"])

        prompt = self.prompt_manager.get(
            "notes_single_agent_prompt", 
            search_query = state["search_query"], 
            research_data = research_data_str
        )

        # Build messages
        messages = [
            SystemMessage(content = prompt)
        ]

        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            return {
                "error": f"NOTES GENERATOR AGENT: LLM call failed: {e}"
            }

        print("\n---------- NOTES GENERATOR AGENT -------------\n")
        print(response)
        print("\n-----------------------\n")

        return {
            "messages": [response]
        }

    # ------------------------------------------------------------------ #

    async def publisher(self, state: NotesGeneratorState):
        """Joins all written sections and renders them to a PDF."""

        full_text = state["messages"][-1].content
        file_path = create_pdf(full_text)
        
        return {"pdf_path": file_path}


# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())