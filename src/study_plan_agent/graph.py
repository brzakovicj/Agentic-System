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
from langgraph.types import Command, RunnableConfig, interrupt
from src.a2a_services.a2a_client import A2A_Client
from src.study_plan_agent.state import StudyPlanState
from src.utils.prompt_manager import PromptManager
from src.study_plan_agent.tools import CourseMatchSchema, handoff_to_agent
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

load_dotenv()

class StudyPlanAgent:
    def __init__(self):
        self._graph = None
        self.agent_cards = ""
        self._prompt_manager = PromptManager()
        self._llm_factory = LLMFactory.get_instance()

        self._tools = [handoff_to_agent]

        self.a2a_client = A2A_Client(
            known_agent_urls=[
                os.getenv("AGENDA_URL"),
            ]
        )

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.mcp_dir = os.path.join(self.base_dir, "mcp")
        self.mcp_config = self.get_config("config.json")

        self.mcp_client = MCPClient(self.mcp_config)
    
    def get_config(self, config_name: str) -> dict:
        """Utility function to load MCP tool configuration from a JSON file."""
        config_path = os.path.join(self.mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            return json.load(f)
        
    def _compress_course_context(
        self,
        course_context: dict,
        max_subtopics_per_topic: int = 4,
        include_hrefs: bool = False,
    ) -> str:
        """
        Converts a raw course_context dict into a compact, LLM-friendly string.
        """
        if not course_context:
            return ""

        course_name = course_context.get("course", "Unknown course")
        topics: list[dict] = course_context.get("topics", [])

        lines: list[str] = [
            f"Course: {course_name}",
            f"Total topics: {len(topics)}",
            "",
        ]

        # De-duplicate topics by (label, title) — the JSON may have duplicates
        seen: set[tuple[str, str]] = set()
        unique_topics: list[dict] = []
        for t in topics:
            key = (t.get("label", ""), t.get("title", ""))
            if key not in seen:
                seen.add(key)
                unique_topics.append(t)

        for idx, topic in enumerate(unique_topics, start=1):
            label    = topic.get("label", f"T{idx}")
            title    = topic.get("title", "Untitled")
            subtopics: list[str] = topic.get("subtopics", [])

            # Skip the first subtopic if it merely echoes the title
            if subtopics and subtopics[0].strip() == title.strip():
                subtopics = subtopics[1:]

            truncated  = subtopics[:max_subtopics_per_topic]
            omitted    = len(subtopics) - len(truncated)
            sub_str    = "; ".join(truncated)
            if omitted > 0:
                sub_str += f"; … (+{omitted} more)"

            href_str = ""
            if include_hrefs:
                href = topic.get("href", "")
                href_str = f" [{href}]" if href else ""

            lines.append(f"{idx}. [{label}] {title}{href_str}")
            if sub_str:
                lines.append(f"   Topics: {sub_str}")

        return "\n".join(lines)

    ##############################################################################################
    # GRAPH BUILDING
    ##############################################################################################

    async def build_graph(self):
        result = await self.a2a_client.a2a_list_discovered_agents()

        self._moodle_tools = await self.mcp_client.get_tools()

        if result["status"] == "success":
            for agent in result["agents"]:
                self.agent_cards += json.dumps(agent, indent=2) + "\n"

        builder = StateGraph(StudyPlanState)

        # Nodes
        builder.add_node("syllabus_matcher", self.syllabus_matcher)
        builder.add_node("syllabus_url", self.syllabus_url)
        builder.add_node("syllabus_loader", self.syllabus_loader)
        builder.add_node("syllabus_loader_tools", ToolNode(self._moodle_tools))

        builder.add_node("study_plan_node", self.study_plan_node)
        builder.add_node("study_plan_tools", ToolNode(self._tools))
        builder.add_node("agenda_agent", self.agenda_agent)

        # Edges
        builder.set_entry_point("syllabus_matcher")

        builder.add_conditional_edges(
            "syllabus_matcher",
            self._route_matcher,
            {
                "found": "study_plan_node",
                "not_found": "syllabus_url",
                "not_found_final": "study_plan_node",
            }
        )

        builder.add_edge("syllabus_url", "syllabus_loader")

        builder.add_conditional_edges(
            "syllabus_loader",
            self._route_loader,
            {
                "tools": "syllabus_loader_tools",
                "done": "syllabus_matcher"
            }
        )

        builder.add_edge("syllabus_loader_tools", "syllabus_loader")

        builder.add_conditional_edges(
            "study_plan_node",
            self._router,
            {
                "tools": "study_plan_tools",
                END: END,
            },
        )

        builder.add_edge("agenda_agent", "study_plan_node")

        self._graph = builder.compile(checkpointer=MemorySaver())
        return self._graph

    async def _router(self, state: StudyPlanState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END
    
    def _route_matcher(self, state: StudyPlanState) -> str:
        if state.get("course_context"):
            return "found"

        if state.get("syllabus_loaded"):
            return "not_found_final"

        return "not_found"
    
    def _route_loader(self, state: StudyPlanState) -> str:
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return "done"

    ##############################################################################################
    # NODES
    ##############################################################################################

    def _load_syllabi(self) -> dict:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_dir))
        syllabi_path = os.path.join(project_root, "course_syllabus/syllabus.json")
        with open(syllabi_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def syllabus_matcher(self, state: StudyPlanState):
        query = state["user_input"] or ""
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

        print("\n--- syllabus_matcher NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)
        print("-----------------------\n")

        return {"course_context": None}
    
    async def syllabus_url(self, state: StudyPlanState):
        llm = self._llm_factory.get_remote_llm()

        system_prompt = self._prompt_manager.get(
            "url_agent",
            user_query=state["user_input"] or ""
        )

        messages = [SystemMessage(content=system_prompt)]

        try:
            response = await llm.ainvoke(messages)
            response = response.content.strip()
            logger.info("LLM URL %s", response)
        except Exception:
            logger.exception("syllabus_url: LLM call failed")
            response = "NONE"

        url: str | None = response if response != "NONE" else None
        
        if not url:
            url = interrupt({
                "message": (
                    "Please send the link to the course syllabus"
                    "(e.g. https://imi.pmf.kg.ac.rs/moodle/course/view.php?id=572)."
                ),
                "type": "text_input",
                "placeholder": "https://imi.pmf.kg.ac.rs/moodle/course/view.php?id=572",
            })

            logger.info("INTERRUPT URL %s", url)

        return {
            "messages": [AIMessage(name="syllabus_url", content="Successfully initialize node.")],
            "syllabus_url": url,
        }

    async def syllabus_loader(self, state: StudyPlanState):
        llm = self._llm_factory.get_tool_llm(
            tier=ModelTier.REMOTE, tools=self._moodle_tools
        )

        tool_context = "\n".join(
            f"{tool.name}: {tool.description}" 
            for tool in self._moodle_tools
        )

        system_prompt = self._prompt_manager.get(
            "syllabus_loader_agent",
            tool_context=tool_context, 
            user_input=state["user_input"] if state["user_input"] else "", 
            course_url=state["syllabus_url"] or "https://imi.pmf.kg.ac.rs/moodle/course/view.php?id=572", 
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            response: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:
            logger.exception(f"StudyPlanAgent exception: {exc}")
            raise

        print("\n--- syllabus_loader NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)
        print("-----------------------\n")

        return {
            "messages": [response],
            "syllabus_loaded": True,
        }

    async def agenda_agent(self, state: StudyPlanState):
        context_id = state.get("agenda_context_id") or str(uuid4())
        message_id = state.get("agenda_message_id") or str(uuid4())
        selected_agent_url = os.getenv("AGENDA_URL")

        got_response = False

        async for result in self.a2a_client.a2a_send_message_stream(
            message_text=state["task_description"] if state["task_description"] else "",
            target_agent_url=selected_agent_url,
            message_id=message_id,
            context_id=context_id,
        ):
            # ERROR
            if result["status"] == "error":
                yield {
                    "messages": [AIMessage(name="agenda", content=result["error"])],
                    "agenda_data": [],
                }
                return
        
            # FINAL
            if result["status"] in ["done", "completed", "success"]:
                response_data = result["response"]

                if response_data["type"] in ["artifact", "message"]:
                    parts = response_data["data"].get("parts", [])
                    if parts:
                        content = parts[0].get("text", "No text response.")
                
                        ai_message = AIMessage(
                            name = "agenda",
                            content = content
                        )

                        yield {
                            "messages": [ai_message],
                            "agenda_data": [ai_message],
                            "agenda_context_id": None,
                            "agenda_message_id": None,
                        }

                        got_response = True

                elif response_data["type"] == "status_update":
                    if response_data.get("state") == "TASK_STATE_INPUT_REQUIRED":
                        message = response_data["data"].get("status", {}).get("message", {})
                        parts = message.get("parts", [])
                        prompt_text = parts[0]["text"] if parts else "Input required."

                        # Sačuvaj context u stanje PRE interrupt-a
                        yield {
                            "agenda_context_id": context_id,
                            "agenda_message_id": message_id,
                        }

                        # Pauziraj graf i pitaj korisnika
                        user_answer = interrupt({
                            "message": prompt_text,
                            "type": "text_input",
                        })

                        # Nastavi sa korisnikovim odgovorom
                        async for inner_result in self.a2a_client.a2a_send_message_stream(
                            message_text=user_answer,
                            target_agent_url=selected_agent_url,
                            message_id=str(uuid4()),
                            context_id=context_id,   # isti context_id!
                        ):
                            if inner_result["status"] in ["done", "completed", "success"]:
                                inner_data = inner_result["response"]
                                if inner_data["type"] in ["artifact", "message"]:
                                    parts = inner_data["data"].get("parts", [])
                                    if parts:
                                        content = parts[0].get("text", "")
                                        ai_message = AIMessage(name="agenda", content=content)
                                        yield {
                                            "messages": [ai_message],
                                            "agenda_data": [ai_message],
                                            "agenda_context_id": None,
                                            "agenda_message_id": None,
                                        }
                                        got_response = True

        if not got_response:
            ai_message = AIMessage(
                name="agenda", 
                content="No response received."
            )

            yield {
                "messages": [ai_message], 
                "agenda_data": []
            }

    async def study_plan_node(self, state: StudyPlanState):
        llm = self._llm_factory.get_tool_llm(
            tier=ModelTier.REMOTE, tools=self._tools
        )

        tool_context = "\n".join(
            f"{tool.name}: {tool.description}" 
            for tool in self._tools
        )

        # course_context_str = self._compress_course_context(
        #     state.get("course_context"),
        #     max_subtopics_per_topic=4,   # tune: lower = fewer tokens
        #     include_hrefs=False,          # enable if LLM needs to cite URLs
        # )

        course_context_str = ""
        if state.get("course_context"):
            course_context_str = json.dumps(state["course_context"], ensure_ascii=False, indent=2)

        logger.info(f"\n--- COMPRESSED COURSE CONTEXT: {course_context_str} ---")

        system_prompt = self._prompt_manager.get(
            "study_plan_agent",
            user_input=state["user_input"] if state["user_input"] else "",
            tool_context=tool_context,
            agent_cards=self.agent_cards,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            course_context=course_context_str,
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

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        config = RunnableConfig(
            recursion_limit=50,
            configurable={
                "thread_id": context_id,
            }
        )

        # ── Determine whether we are resuming an interrupted run ──────────
        snapshot = await self._graph.aget_state(config)
        is_resuming = bool(snapshot.next)

        if is_resuming:
            state: Any = Command(resume=query)
            logger.debug(
                "Resuming thread=%s next=%s",
                context_id,
                snapshot.next,
            )
        else:
            # Fresh execution
            state = StudyPlanState(
                messages=[HumanMessage(content=query)],
                user_input=query,
                task_description=None,
                agenda_data=[],
                course_context=None,
                agenda_context_id=None,
                agenda_message_id=None,
                syllabus_loaded=False,
                syllabus_url=None,
            )

        last_ai_content = ""
        completed_normally = False
        interrupted = False

        logger.debug("astream called: is_resuming=%s, query=%r", is_resuming, query)

        try:
            async for item in self._graph.astream(
                input = state,
                stream_mode="updates",
                config = config
            ):
                if "__interrupt__" in item:
                    interrupted = True

                    for interrupt_obj in item["__interrupt__"]:
                        iv = interrupt_obj.value

                        yield {
                            "is_task_complete": False,
                            "require_user_input": True,
                            "content": iv["message"] if iv["message"] else "Input required.",
                        }
                else:
                    # updates je dict: {node_name: {"messages": [...]}}
                    for node_name, state_update in item.items():
                        messages = state_update.get("messages", [])

                        for msg in messages:
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(f"  TOOL CALL [{node_name}]: {tc['name']}")
                                        print(f"  {tc['args']}")
                                    print()

                                    yield {
                                        'is_task_complete': False,
                                        'require_user_input': False,
                                        'content': 'Calling tools...',
                                    }

                                elif msg.content:
                                    last_ai_content = msg.content.strip()
                                    yield {
                                        'is_task_complete': False,
                                        'require_user_input': False,
                                        'content': last_ai_content,
                                    }

                            elif isinstance(msg, ToolMessage):
                                print(
                                    f"[Tool result: {msg.name}]"
                                )
                                print()
                                
                                if (isinstance(msg.content, list)):
                                    tool_parts = msg.content
                                    msg_content = "\n\n".join(
                                        m.content for m in tool_parts
                                        if hasattr(m, "content") and m.content
                                    )
                                else:
                                    msg_content = msg.content or "[Tool returned no content]"
                                
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': 'Tool responded with results: \n' + msg_content,
                                }

            completed_normally = True

        except Exception as exc:
            print(f"Graph execution failed: {exc}")

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(exc)}",
            }

        finally:
            if completed_normally and not interrupted:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_ai_content,
                }