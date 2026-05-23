import json
import os
import logging
from typing import Any, AsyncIterable
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from dotenv import load_dotenv
from langgraph.types import RunnableConfig
from src.a2a_services.a2a_client import A2A_Client
from src.study_plan_agent.state import StudyPlanState
from src.prompts.prompt_manager import PromptManager
from src.study_plan_agent.tools import handoff_to_agent
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

load_dotenv()

class StudyPlanAgent:
    def __init__(self):
        self._graph = None
        self._mcp_tools = None
        self.agent_cards = ""
        self._prompt_manager = PromptManager()
        self._llm_factory = LLMFactory.get_instance()

        self._tools = [handoff_to_agent]

        self.a2a_client = A2A_Client(
            known_agent_urls=[
                os.getenv("SCHOLAR_URL"),
                os.getenv("AGENDA_URL"),
            ]
        )

    ##############################################################################################
    # HELPER FUNCTIONS
    ##############################################################################################

    def _get_config(self, config_name: str) -> dict:
        config_path = os.path.join(self._mcp_dir, config_name)
        with open(config_path, "r") as f:
            return json.load(f)

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def build_graph(self):
        result = await self.a2a_client.a2a_list_discovered_agents()

        if result["status"] == "success":
            for agent in result["agents"]:
                self.agent_cards += json.dumps(agent, indent=2) + "\n"

        builder = StateGraph(StudyPlanState)

        # Nodes
        builder.add_node("study_plan_node", self.study_plan_node)
        builder.add_node("study_plan_tools", ToolNode(self._tools))
        builder.add_node("execute_agent", self.execute_agent)

        # Edges
        builder.set_entry_point("study_plan_node")

        builder.add_conditional_edges(
            "study_plan_node",
            self._router,
            {
                "tools": "study_plan_tools",
                END: END,
            },
        )

        builder.add_edge("execute_agent", "study_plan_node")

        self._graph = builder.compile(checkpointer=MemorySaver())
        return self._graph

    async def _router(self, state: StudyPlanState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    ##############################################################################################
    # NODES
    ##############################################################################################

    async def execute_agent(self, state: StudyPlanState):
        
        selected_agent = state["selected_agent"] or ""

        if selected_agent == "":
            ai_message = AIMessage(
                name = "error",
                content = "Agent is not selected."
            )

            yield {
                "messages": [ai_message],
                "agenda_data": [],
                "scholar_data": []
            }
            return
        
        if selected_agent == "agenda":
            selected_agent_url = os.getenv("AGENDA_URL")
        elif selected_agent == "scholar":
            selected_agent_url = os.getenv("SCHOLAR_URL")

        message_id = str(uuid4())

        got_response = False
        async for result in self.a2a_client.a2a_send_message_stream(
            message_text=state["task_description"] if state["task_description"] else "",
            target_agent_url=selected_agent_url,
            message_id=message_id,
        ):
            # ERROR
            if result["status"] == "error":
                ai_message = AIMessage(
                    name = selected_agent,
                    content = result['error']
                )

                if selected_agent == "agenda":
                    yield {
                        "messages": [ai_message],
                        "agenda_data": []
                    }
                elif selected_agent == "scholar":
                    yield {
                        "messages": [ai_message],
                        "scholar_data": []
                    }
                return 
            # FINAL
            if result["status"] in ["done", "completed", "success"]:
                response_data = result["response"]

                if response_data["type"] == "artifact":
                    parts = response_data["data"].get("parts", [])
                    if parts:
                        content = parts[0].get("text", "No text response.")
                
                        ai_message = AIMessage(
                            name = selected_agent,
                            content = content
                        )

                        if selected_agent == "agenda":
                            yield {
                                "messages": [ai_message],
                                "agenda_data": [ai_message]
                            }
                        elif selected_agent == "scholar":
                            yield {
                                "messages": [ai_message],
                                "scholar_data": [ai_message]
                            }
                        got_response = True

                elif response_data["type"] == "message":
                    parts = response_data["data"].get("parts", [])
                    if parts:
                        content = parts[0].get("text", "No text response.")

                        ai_message = AIMessage(
                            name = selected_agent,
                            content = content
                        )

                        if selected_agent == "agenda":
                            yield {
                                "messages": [ai_message],
                                "agenda_data": [ai_message]
                            }
                        elif selected_agent == "scholar":
                            yield {
                                "messages": [ai_message],
                                "scholar_data": [ai_message]
                            }

        if not got_response:
            ai_message = AIMessage(
                name=selected_agent, 
                content="No response received."
            )
            
            if selected_agent == "agenda":
                yield {"messages": [ai_message], "agenda_data": [ai_message]}
            elif selected_agent == "scholar":
                yield {"messages": [ai_message], "scholar_data": [ai_message]}

    async def study_plan_node(self, state: StudyPlanState):
        llm = self._llm_factory.get_tool_llm(
            tier=ModelTier.REMOTE, tools=self._tools
        )

        tool_context = "\n".join(
            f"{tool.name}: {tool.description}" 
            for tool in self._tools
        )

        system_prompt = self._prompt_manager.get(
            "study_plan_agent",
            user_input=state["user_input"] if state["user_input"] else "",
            tool_context=tool_context,
            agent_cards=self.agent_cards,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            response: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:
            logger.error(f"StudyPlanAgent exception: {exc}")
            raise

        return {"messages": [response]}

    ##############################################################################################
    # STREAM
    ##############################################################################################

    async def astream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        state = StudyPlanState(
            messages=[HumanMessage(content=query)],
            user_input=query,
            task_description=None,
            selected_agent=None,
            scholar_data=[],
            agenda_data=[]
        )

        config = RunnableConfig(
            recursion_limit=10,
            configurable={
                "thread_id": context_id,
            }
        )

        last_ai_content = ""
        completed_normally = False

        try:
            async for item in self._graph.astream(
                input=state,
                stream_mode="updates",
                config=config,
            ):
                for node_name, state_update in item.items():
                    messages = state_update.get("messages", [])

                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": "Pristupam task menadžeru...",
                                }

                            elif msg.content:
                                last_ai_content = msg.content.strip()
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": last_ai_content,
                                }

                        elif isinstance(msg, ToolMessage):
                            if isinstance(msg.content, list):
                                msg_content = "\n\n".join(
                                    m.content
                                    for m in msg.content
                                    if hasattr(m, "content") and m.content
                                )
                            else:
                                msg_content = msg.content or "[Tool returned no content]"

                            yield {
                                "is_task_complete": False,
                                "require_user_input": False,
                                "content": f"Rezultat alata ({msg.name}):\n{msg_content}",
                            }

            completed_normally = True

        except Exception as exc:
            logger.error(f"Greška u izvršavanju grafa: {exc}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Greška: {str(exc)}",
            }

        finally:
            if completed_normally:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_ai_content,
                }