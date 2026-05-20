import json
import os
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

load_dotenv()

class StudyPlanAgent:
    def __init__(self):
        self._graph = None
        self._mcp_tools = None
        self.agent_cards = ""
        self._prompt_manager = PromptManager()
        self._llm_factory = LLMFactory.get_instance()

        # _base_dir = os.path.dirname(os.path.abspath(__file__))
        # self._mcp_dir = os.path.join(_base_dir, "mcp")
        # self._config = self._get_config("config.json")

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
        builder.add_node("scholar_agent", self.scholar_agent)
        builder.add_node("agenda_agent", self.agenda_agent)

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

        builder.add_edge("scholar_agent", "study_plan_node")
        builder.add_edge("agenda_agent", "study_plan_node")

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

    async def scholar_agent(self, state: StudyPlanState):

        selected_agent = os.getenv("SCHOLAR_URL")

        message_id = str(uuid4())

        result = await self.a2a_client.a2a_send_message(
            message_text=state["task_description"] if state["task_description"] else "",
            target_agent_url=selected_agent,
            message_id=message_id,
        )

        if result["status"] == "error":
            ai_message = AIMessage(
                name = "scholar",
                content = result['error']
            )

            return {
                "messages": [ai_message],
                "scholar_data": []
            }

        response_data = result["response"]

        # artifact
        if response_data["type"] == "artifact":

            artifact = response_data["data"]

            parts = artifact.get("parts", [])

            if parts:
                content = parts[0].get("text", "No text response.")
                
                ai_message = AIMessage(
                    name = "scholar",
                    content = content
                )

                return {
                    "messages": [ai_message],
                    "scholar_data": [ai_message]
                }

        # direct message
        if response_data["type"] == "message":

            message = response_data["data"]

            parts = message.get("parts", [])

            if parts:
                content = parts[0].get("text", "No text response.")

                ai_message = AIMessage(
                    name = "scholar",
                    content = content
                )

                return {
                    "messages": [ai_message],
                    "scholar_data": [ai_message]
                }

        ai_message = AIMessage(
            name = "scholar",
            content = "No response received."
        )

        return {
            "messages": [ai_message],
            "scholar_data": [ai_message]
        }

    async def agenda_agent(self, state: StudyPlanState):
        
        selected_agent = os.getenv("AGENDA_URL")

        message_id = str(uuid4())

        result = await self.a2a_client.a2a_send_message(
            message_text=state["task_description"] if state["task_description"] else "",
            target_agent_url=selected_agent,
            message_id=message_id,
        )

        if result["status"] == "error":
            ai_message = AIMessage(
                name = "agenda",
                content = result['error']
            )

            return {
                "messages": [ai_message],
                "agenda_data": []
            }

        response_data = result["response"]

        # artifact
        if response_data["type"] == "artifact":

            artifact = response_data["data"]

            parts = artifact.get("parts", [])

            if parts:
                content = parts[0].get("text", "No text response.")
                
                ai_message = AIMessage(
                    name = "agenda",
                    content = content
                )

                return {
                    "messages": [ai_message],
                    "agenda_data": [ai_message]
                }

        # direct message
        if response_data["type"] == "message":

            message = response_data["data"]

            parts = message.get("parts", [])

            if parts:
                content = parts[0].get("text", "No text response.")

                ai_message = AIMessage(
                    name = "agenda",
                    content = content
                )

                return {
                    "messages": [ai_message],
                    "agenda_data": [ai_message]
                }

        ai_message = AIMessage(
            name = "agenda",
            content = "No response received."
        )

        return {
            "messages": [ai_message],
            "agenda_data": [ai_message]
        }

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
            print(f"StudyPlanAgent exception: {exc}")
            raise

        print("\n--- STUDY PLAN NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)
        print("-----------------------\n")

        return {"messages": [response]}

    ##############################################################################################
    # STREAM
    ##############################################################################################

    async def astream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        state = StudyPlanState(
            messages=[HumanMessage(content=query)],
            user_input=query,
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
                                for tc in msg.tool_calls:
                                    print(f"  TOOL CALL [{node_name}]: {tc['name']}")
                                    print(f"  {tc['args']}")

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
                            print(f"[Tool result: {msg.name}]")

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
            print(f"Greška u izvršavanju grafa: {exc}")
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