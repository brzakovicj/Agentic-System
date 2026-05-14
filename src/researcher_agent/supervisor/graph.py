from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from dotenv import load_dotenv
from langgraph.types import RunnableConfig
from src.researcher_agent.researcher.graph import ResearcherAgent
from src.researcher_agent.notes_generator.graph import NotesGeneratorAgent
from src.researcher_agent.supervisor.state import SupervisorState
from src.researcher_agent.supervisor.tools import PlannerTaskSchema, handoff_to_subagent
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier

from tenacity import RetryError

from src.utils.retryable_invoke import ainvoke_llm

load_dotenv()

class SupervisorAgent:
    def __init__(self):
        self.graph = None
        self.tools = None
        self._prompt_manager = PromptManager()
        
        self.tools = [handoff_to_subagent]

        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self.tools
        )

        self.supervisor_prompt = self._prompt_manager.get("supervisor", tools=tools_context, current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self._llm_factory = LLMFactory.get_instance()
        self.llm_with_tools = self._llm_factory.get_tool_llm(ModelTier.REMOTE, self.tools)

        self.research_agent = ResearcherAgent()
        self.notes_generator = NotesGeneratorAgent()

    async def build_graph(self):
        """Kreira LangGraph."""
        builder = StateGraph(SupervisorState)

        self.research_graph = await self.research_agent.initialize()
        self.notes_generator_graph = await self.notes_generator.initialize()

        builder.add_node("planner", self.planner)
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node(self.call_researcher)
        builder.add_node("call_notes_generator",self.call_notes_generator)

        builder.set_entry_point("planner")

        builder.add_edge("planner", "supervisor")

        builder.add_conditional_edges(
            "supervisor",
            self.supervisor_router,
            {
                "tools": "tools",
                END: END,
            }
        )

        builder.add_edge("call_researcher", "supervisor")
        builder.add_edge("call_notes_generator", "supervisor")

        self.graph = builder.compile(checkpointer=MemorySaver())

        return self.graph
    
    def supervisor_router(self, state: SupervisorState) -> str:
        """Route to the tools node if the supervisor makes a tool call."""
        last_message = state["messages"][-1]

        if state["final_answer"]:
            return END

        if last_message.tool_calls:
            return "tools"
        
        return END
    
    ###################################################################################
    # NODE FUNCTIONS
    ###################################################################################

    async def planner(self, state: SupervisorState):
        """The planner agent, which breaks down the task into smaller sub-tasks."""
        llm = self._llm_factory.get_llm_with_structured_output(
            schema=PlannerTaskSchema, 
            tier=ModelTier.REMOTE
        )

        # llm = self._llm_factory.get_remote_llm()

        prompt = self._prompt_manager.get(
            "task_planner_prompt", 
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            user_request = state["messages"][-1].content if state["messages"] else "None")
        messages = [SystemMessage(content=prompt)] + state["messages"]

        try:
            response = await llm.ainvoke(messages)

            print("\n--- PLANNER NODE ---\n")
            print(response)
            print("\n-----------------------\n")

            return {
                "plan": response["plan"],
                "current_task_idx": 0,
            }
        except Exception as e:
            print(f"Error in planner node: {e}")
            raise
    
    async def supervisor(self, state: SupervisorState):
        """The main supervisor agent."""
        idx = state["current_task_idx"]
        plan = state["plan"] if state["plan"] else []

        if not plan or idx >= len(plan):
            return {
                "final_answer": True,
            }
        
        total = len(plan)
        current_task = plan[idx]

        # Provide only the two most recent messages as context to avoid
        # inflating the prompt with the entire messages on every iteration.
        recent = state["messages"][-2:] if state["messages"] else []
        messages = "\n\n".join(
            m.content for m in recent
            if hasattr(m, "content") and m.content
        )

        prompt = self._prompt_manager.get(
            "supervisor",
            idx = idx + 1,
            total = total,
            task_name = current_task["name"],
            task_description = current_task["description"],
            recent_messages = messages if messages else "None yet.",
        )

        try:
            response = await ainvoke_llm(self.llm_with_tools, [SystemMessage(content = prompt)])
        except RetryError as e:
            print("LLM call failed after all retries: %s", e.last_attempt.exception())
            raise
        except Exception:
            print("Exception: Non-retryable LLM error")
            raise

        print("\n--- SUPERVISOR NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        return {
            "messages": [response],
            "current_task_idx": idx + 1,
        }

    async def call_notes_generator(self, state: SupervisorState, config: RunnableConfig):
        """Call the notes generator agent.
        
        The agent is invoked with the task description generated by the supervisor, and any research reports that have been generated by the researcher.
        """
        notes_input = {
            "messages": [HumanMessage(content=state["task_description"])],
            "search_query": state["task_description"],
            "research_data": state["research_data"] if state["research_data"] else "",
            "pdf_path": "",
        }

        result = await self.notes_generator_graph.ainvoke(
            input = notes_input,
            config = config,
        )

        pdf_path = result["pdf_path"] if result["pdf_path"] else "unknown location"

        ai_message = AIMessage(
            content=f"Study script generated successfully.\n Saved to: {pdf_path}"
        )

        return {
            "messages": [ai_message],
        }

    async def call_researcher(self, state: SupervisorState, config: RunnableConfig):
        """Call the researcher agent.
        
        The agent is invoked with the task description generated by the supervisor, and any research reports that have been generated by the researcher.
        """
        research_response = await self.research_graph.ainvoke(
            input={
                "messages": [HumanMessage(content=state["task_description"])],
                "query": state["task_description"],
                },
            config=config,
        )

        print("\n--- RESEARCHER STATE ---\n")
        print(type(research_response).__name__, getattr(research_response["messages"][-1], "content", ""))
        if hasattr(research_response["messages"][-1], "tool_calls"):
            print("TOOL CALLS:", research_response["messages"][-1].tool_calls)

        ai_message = AIMessage(
            name="researcher", 
            content=research_response['messages'][-1].content
        )

        return {
            "messages": [ai_message],
            "research_data": [research_response['messages'][-1].content],
        }

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())