import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from src.utils.mcp_client import MCPClient
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.multi_agent.researcher.state import ResearcherState
from src.multi_agent.researcher.tools import EvaluatorDecision

load_dotenv()

class ResearcherAgent:
    def __init__(self):
        self._graph = None
        self._prompt_manager = PromptManager()
        self.factory = LLMFactory.get_instance()
        
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._mcp_dir = os.path.join(_base_dir, "mcp")

    async def initialize(self):
        retriever_config = self._get_config("retriever_config.json")
        web_searcher_config = self._get_config("web_searcher_config.json")
        
        # Cleanly replace any previous MCP client to avoid connection leaks
        if hasattr(self, "mcp_client"):
            await self.mcp_client.close()

        mcp_client = MCPClient(retriever_config)
        self._mcp_tools_retriever_node = await mcp_client.get_tools()

        mcp_client = MCPClient(web_searcher_config)
        self._mcp_tools_web_searcher_node = await mcp_client.get_tools()

        await self._build_graph()

        return self._graph

    ##############################################################################################
    # HEPLER FUNCTIONS
    ##############################################################################################

    def _get_config(self, config_name: str) -> dict:
        """Utility function to load MCP tool configuration from a JSON file."""
        config_path = os.path.join(self._mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            return json.load(f)

    def _stringify_content(self, content) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, (dict, list)):
            try:
                return json.dumps(
                    content,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:
                return str(content)

        return str(content)

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def _build_graph(self):
        """Build the state graph for the ResearcherAgent."""
        builder = StateGraph(ResearcherState)

        builder.add_node("retriever", self.retriever_node)
        builder.add_node("retriever_tools", ToolNode(self._mcp_tools_retriever_node))
        builder.add_node("evaluator", self.evaluator_node)
        builder.add_node("web_searcher", self.web_searcher_node)
        builder.add_node("web_searcher_tools", ToolNode(self._mcp_tools_web_searcher_node))
        builder.add_node("synthesizer", self.synthesizer_node)

        builder.set_entry_point("retriever")

        builder.add_conditional_edges(
            "retriever",
            self.retriever_router,
            {
                "tools": "retriever_tools",
                "evaluator": "evaluator",
            }
        )

        builder.add_edge("retriever_tools", "retriever")

        builder.add_conditional_edges(
            "evaluator",
            self.evaluator_router,
            {
                "web_searcher": "web_searcher",
                "synthesizer": "synthesizer",
            }
        )

        builder.add_conditional_edges(
            "web_searcher",
            self.web_searcher_router,
            {
                "tools": "web_searcher_tools",
                "synthesizer": "synthesizer",
            }
        )

        builder.add_edge("web_searcher_tools", "web_searcher")

        builder.add_edge("synthesizer", END)

        # Don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
        self._graph = builder.compile()

    async def retriever_router(self, state: ResearcherState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "evaluator"

    async def evaluator_router(self, state: ResearcherState) -> str:
        """Route to the web search node if the evaluator determines that retrieval was insufficient, otherwise route to the sintesis node."""
        if state["need_web_search"]:
            return "web_searcher"
        return "synthesizer"
    
    async def web_searcher_router(self, state: ResearcherState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "synthesizer"

    ##############################################################################################
    # GRAPH NODES
    ##############################################################################################

    async def retriever_node(self, state: ResearcherState):
        """
            A node that retrieves relevant documents based on the research query. 
            If the retrieval is successful and returns relevant documents, the node will 
            return those documents in the state.
        """
        # LLM
        llm = self.factory.get_tool_llm(tier=ModelTier.REMOTE, tools=self._mcp_tools_retriever_node)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._mcp_tools_retriever_node
        )

        research_retriever_prompt = self._prompt_manager.get("research_retriever_prompt", query=state["query"], tools=tools_context)
        
        # Build messages
        messages = [
            SystemMessage(content=research_retriever_prompt)
        ]

        try:
            response = await llm.ainvoke(messages + state["messages"])
        except Exception as e:
            return {
                "error": f"LLM call failed: {e}"
            }
        
        print("\n--- RETRIEVER STATE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")
        
        return {
            "messages": [response],
            "retrieved_docs": [response.content] if not response.tool_calls else []
        }

    async def evaluator_node(self, state: ResearcherState) -> dict:
        """
            A node that evaluates the relevance and sufficiency of the retrieved documents for answering the research query.
            The node will take the retrieved documents and the original query as input and determine whether the retrieved 
            information is sufficient to answer the research question or if a web search is needed to gather more information.
            The node will return a boolean value in the state indicating whether a web search is needed or not. 
            If there are errors during evaluation, it will return an error message and default to indicating that a web 
            search is needed.
        """
        retrieval_results = state["retrieved_docs"] or []

        if not retrieval_results or len(retrieval_results) == 0:
            return {
                "need_web_search": True,
            }

        # LLM
        llm = self.factory.get_llm_with_structured_output(schema=EvaluatorDecision, tier=ModelTier.REMOTE)

        docs_context = "\n\n".join(
            f"{self._stringify_content(doc)}"
            for doc in retrieval_results
        )

        evaluator_prompt = self._prompt_manager.get("research_evaluator_prompt", query=state["query"], documents=docs_context)

        messages = [
            SystemMessage(content=evaluator_prompt)
        ]
 
        try:
            decision = await llm.ainvoke(messages)

            print("\n---------- EVALUATOR-------------\n")
            print(decision)
            print("\n-----------------------\n")

            return {
                "need_web_search": decision["need_web_search"]
            }
        except Exception as exc:
            return {
                "need_web_search": True, 
                "error": str(exc)
            }

    async def web_searcher_node(self, state: ResearcherState) -> dict:
        """
            A node that performs a web search if the evaluator determines that the retrieved documents are insufficient.
            The node will use the research query to perform a web search and return the results in the state. 
            If the web search is successful, the results will be returned in the state. 
            If there are errors during the web search, it will return an error message and an empty list of results.
        """
        # LLM
        llm = self.factory.get_tool_llm(tier=ModelTier.REMOTE, tools=self._mcp_tools_web_searcher_node)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._mcp_tools_web_searcher_node
        )

        web_prompt = self._prompt_manager.get(
            "research_web_searcher_prompt",
            query=state["query"],
            tools=tools_context,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # messages
        messages = [
            SystemMessage(content=web_prompt)
        ]
 
        try:
            response = await llm.ainvoke(messages + state["messages"])
        except Exception as exc:
            return {
                "error": str(exc)
            }
        
        print("\n--- WEB SEARCHER STATE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")

        return {
            "messages": [response],
            "web_results": [response.content] if not response.tool_calls else []
        }

    async def synthesizer_node(self, state: ResearcherState) -> dict:
        """
            A node that synthesizes the final research report based on the retrieved documents and web search results.
            The node will take all the retrieved information and generate a comprehensive answer to the research query.
            If the synthesis is successful, the node will return the final answer in the state. If there are errors during synthesis, 
            it will return an error message and a fallback answer.
            The synthesizer should also be able to handle cases where the retrieved information is insufficient and 
            explicitly state that in the final answer if that's the case.
        """
        # LLM
        llm = self.factory.get_base_llm()
        
        retrieved_sources = state["retrieved_docs"] or []
        web_sources = state["web_results"] or []

        all_sources = "\n\n".join(
            f"{self._stringify_content(source)}"
            for source in retrieved_sources + web_sources
        )
        
        prompt = self._prompt_manager.get(
            "research_synthesizer_prompt",
            query=state["query"],
            sources=all_sources,
        )
 
        messages = [
            SystemMessage(content=prompt)
        ]
 
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            return {
                "error": str(exc),
                "final_answer": "I'm sorry, I was not able to synthesize the research findings due to an error."
            }

        print("\n--- SYNTHESIZER STATE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")

        return {
            "final_answer": response.content,
            "messages": [response]
        }
    

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())