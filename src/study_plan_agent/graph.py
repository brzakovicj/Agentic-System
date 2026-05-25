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
from src.study_plan_agent.tools import CourseMatchSchema, handoff_to_agent
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

        self._tools = [handoff_to_agent]

        self.a2a_client = A2A_Client(
            known_agent_urls=[
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
        builder.add_node("syllabus_loader", self.syllabus_loader)  # NEW
        builder.add_node("study_plan_node", self.study_plan_node)
        builder.add_node("study_plan_tools", ToolNode(self._tools))
        builder.add_node("execute_agent", self.execute_agent)

        # Edges
        builder.set_entry_point("syllabus_loader")                 # CHANGED
        builder.add_edge("syllabus_loader", "study_plan_node")     # NEW

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

    def _load_syllabi(self) -> dict:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_dir))
        syllabi_path = os.path.join(project_root, "course_syllabus/syllabus.json")
        with open(syllabi_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def syllabus_loader(self, state: StudyPlanState):
        query = state["messages"][-1].content if state["messages"] else ""
        syllabi = self._load_syllabi()
        courses = syllabi["courses"]

        llm = self._llm_factory.get_llm_with_structured_output(
            schema=CourseMatchSchema,
            tier=ModelTier.REMOTE
        )

        prompt = self._prompt_manager.get(
            "syllabus_agent", 
            query = query, 
            courses = json.dumps([{"key": k, "course": v["course"]} for k, v in courses.items()], ensure_ascii=False)
        )

        try:
            response = await llm.ainvoke([SystemMessage(content=prompt)])
            key = response["matched_key"]
            if key and key in courses:
                matched = courses[key]
                print(f"\n--- STUDY PLAN SYLLABUS LOADER: matched '{matched['course']}' ---\n")
                return {"course_context": matched}
        except Exception as e:
            print(f"Study plan syllabus loader error: {e}")

        return {"course_context": None}

    async def execute_agent(self, state: StudyPlanState):
        
        selected_agent = state["selected_agent"] or ""

        if selected_agent == "":
            ai_message = AIMessage(
                name = "error",
                content = "Agent is not selected."
            )

            yield {
                "messages": [ai_message],
                "agenda_data": []
            }
            return
        
        if selected_agent == "agenda":
            selected_agent_url = os.getenv("AGENDA_URL")

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

        if not got_response:
            ai_message = AIMessage(
                name=selected_agent, 
                content="No response received."
            )
            
            if selected_agent == "agenda":
                yield {"messages": [ai_message], "agenda_data": [ai_message]}

    async def study_plan_node(self, state: StudyPlanState):
        llm = self._llm_factory.get_tool_llm(
            tier=ModelTier.REMOTE, tools=self._tools
        )

        tool_context = "\n".join(
            f"{tool.name}: {tool.description}" 
            for tool in self._tools
        )

        course_context_str = ""
        if state.get("course_context"):
            course_context_str = json.dumps(state["course_context"], ensure_ascii=False, indent=2)

        system_prompt = self._prompt_manager.get(
            "study_plan_agent",
            user_input=state["user_input"] if state["user_input"] else "",
            tool_context=tool_context,
            agent_cards=self.agent_cards,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            course_context=course_context_str,  # NEW
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
            task_description=None,
            selected_agent=None,
            agenda_data=[],
            course_context=None,  # NEW
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