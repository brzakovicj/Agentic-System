from typing import Any, AsyncIterable

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from dotenv import load_dotenv
from langgraph.types import RunnableConfig
from tenacity import RetryError
from src.agenda_agent.agenda.state import AgendaState
from src.agenda_agent.agenda.tools import PlannerTaskSchema, handoff_to_subagent
from src.agenda_agent.scheduler.graph import SchedulerAgent
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.retryable_invoke import ainvoke_llm

load_dotenv()

class AgendaAgent:
    def __init__(self):
        self._graph = None
        self._tools = None
        self._prompt_manager = PromptManager()
        
        self._tools = [handoff_to_subagent]

        self._llm_factory = LLMFactory.get_instance()
        self._llm_with_tools = self._llm_factory.get_tool_llm(ModelTier.REMOTE, self._tools)

        self._scheduler_agent = SchedulerAgent()

    async def build_graph(self):
        builder = StateGraph(AgendaState)

        self._scheduler_graph = await self._scheduler_agent.initialize()

        builder.add_node("planner", self.planner)
        builder.add_node("orchestrator", self.orchestrator)
        builder.add_node("tools", ToolNode(self._tools))
        builder.add_node("call_scheduler", self.call_scheduler)

        builder.set_entry_point("planner")

        builder.add_edge("planner", "orchestrator")

        builder.add_conditional_edges(
            "orchestrator",
            self.orchestrator_router,
            {
                "tools": "tools",
                END: END,
            }
        )

        builder.add_edge("call_scheduler", "orchestrator")

        self._graph = builder.compile(checkpointer=MemorySaver())

        return self._graph
    
    def orchestrator_router(self, state: AgendaState) -> str:
        """Route to the tools node if the orchestrator makes a tool call."""
        last_message = state["messages"][-1]

        if state["final_answer"]:
            return END

        if last_message.tool_calls:
            return "tools"
        
        return END
    
    ###################################################################################
    # NODE FUNCTIONS
    ###################################################################################

    async def planner(self, state: AgendaState):
        """The planner agent, which breaks down the task into smaller sub-tasks."""
        llm = self._llm_factory.get_llm_with_structured_output(
            schema=PlannerTaskSchema, 
            tier=ModelTier.REMOTE
        )

        prompt = self._prompt_manager.get(
            "task_planner_prompt",
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_request=state["messages"][-1].content if state["messages"] else ""
        )
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
    
    async def orchestrator(self, state: AgendaState):
        """The main orchestrator agent."""
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
            "orchestrator",
            idx = idx + 1,
            total = total,
            task_name = current_task["name"],
            task_description = current_task["description"],
            recent_messages = messages if messages else "None yet.",
        )

        try:
            response = await ainvoke_llm(self._llm_with_tools, [SystemMessage(content = prompt)])
        except RetryError as e:
            print("LLM call failed after all retries: %s", e.last_attempt.exception())
            raise
        except Exception:
            print("Exception: Non-retryable LLM error")
            raise

        print("\n--- ORCHESTRATOR NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        return {
            "messages": [response],
            "current_task_idx": idx + 1,
        }

    async def call_scheduler(self, state: AgendaState, config: RunnableConfig):
        """Call the scheduler agent.
        
        The agent is invoked with the task description generated by the orchestrator, and any research reports that have been generated by the researcher.
        """
        scheduler_response = await self._scheduler_graph.ainvoke(
            input={
                "messages": [HumanMessage(content=state["task_description"])],
                },
            config=config,
        )

        print("\n--- SCHEDULER STATE ---\n")
        print(type(scheduler_response).__name__, getattr(scheduler_response["messages"][-1], "content", ""))
        if hasattr(scheduler_response["messages"][-1], "tool_calls"):
            print("TOOL CALLS:", scheduler_response["messages"][-1].tool_calls)

        ai_message = AIMessage(
            name="scheduler", 
            content=scheduler_response['messages'][-1].content
        )

        return {
            "messages": [ai_message],
        }
    
    ############################### STREAM ####################################

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        state = AgendaState(
            messages = [ HumanMessage(content = query) ],
            final_answer = False
        )

        config = RunnableConfig(configurable={
            "thread_id": context_id,
            "recursion_limit": 50,
        })

        async for item in self._graph.astream(
            input = state,
            stream_mode="updates",
            subgraphs=True,
            config = config
        ):
            # Sa subgraphs=True i "updates" struktura je: (namespace, {node_name: state_update})
            namespace, updates = item

            # updates je dict: {node_name: {"messages": [...]}}
            for node_name, state_update in updates.items():
                messages = state_update.get("messages", [])
                is_final = state_update.get("final_answer", False)

                for msg in messages:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  🔧 TOOL CALL [{node_name}]: {tc['name']}")
                                print(f"  {tc['args']}")
                            print()

                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': 'Calling tools...',
                            }

                        elif is_final:
                            yield {
                                'is_task_complete': True,
                                'require_user_input': False,
                                'content': msg.content.strip(),
                            }

                        elif msg.content:
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': msg.content.strip(),
                            }

                    elif isinstance(msg, ToolMessage):
                        print(
                            f"[Tool result: {msg.name}]"
                        )
                        print()
                        
                        if (isinstance(msg.content, list)):
                            messages = msg.content
                            msg_content = "\n\n".join(
                                m.content for m in messages
                                if hasattr(m, "content") and m.content
                            )
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': 'Tool responded with results: \n' + msg_content,
                            }
                        else:
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': 'Tool responded with results: \n' + msg.content,
                            }
                
                if is_final and not messages:
                    yield {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': 'Task completed.',
                    }


#Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())