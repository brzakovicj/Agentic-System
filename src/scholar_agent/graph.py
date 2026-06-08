import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, AsyncIterable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig
from src.scholar_agent.state import ScholarState
from src.scholar_agent.tools import CourseMatchSchema, PlannerTaskSchema, create_pdf, handoff_to_subagent
from src.utils.tool_formatter import llm_describe_tool_call
from src.utils.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient

load_dotenv()

class ScholarAgent:
    def __init__(self):
        self.graph = None
        self.tools = None
        self.prompt_manager = PromptManager()
        
        self.tools = [handoff_to_subagent]

        self.llm_factory = LLMFactory.get_instance()
        self.llm_with_tools = self.llm_factory.get_tool_llm(ModelTier.REMOTE, self.tools)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.mcp_dir = os.path.join(self.base_dir, "mcp")
        self.mcp_config = self.get_config("config.json")

        self.mcp_client = MCPClient(self.mcp_config)

    async def build_graph(self):
        """Kreira LangGraph."""
        builder = StateGraph(ScholarState)

        # MCP tools for researcher
        self.mcp_tools = await self.mcp_client.get_tools()

        # ----- NODES -----

        builder.add_node("syllabus_loader", self.syllabus_loader)
        builder.add_node("planner", self.planner)
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("supervisor_tools", ToolNode(self.tools))

        # Researcher
        builder.add_node("researcher_node", self.researcher_node)
        builder.add_node("researcher_tools", ToolNode(self.mcp_tools, messages_key="researcher_messages"))
        builder.add_node("researcher_done", self.researcher_done)
        builder.add_node("combine_research", self.combine_research)
        
        # Notes generator
        builder.add_node("notes_node", self.notes_node)
        builder.add_node("notes_done", self.notes_done)

        # ----- EDGES -----

        builder.set_entry_point("syllabus_loader")
        builder.add_edge("syllabus_loader", "planner")
        builder.add_edge("planner", "supervisor")

        builder.add_conditional_edges(
            "supervisor",
            self.supervisor_router,
            {
                "supervisor_tools": "supervisor_tools",
                "combine_research": "combine_research",
                END: END,
            }
        )

        builder.add_conditional_edges(
            "researcher_node",
            self.researcher_router,
            {
                "researcher_tools": "researcher_tools",
                END: "researcher_done",
            }
        )

        builder.add_edge("researcher_tools", "researcher_node")
        builder.add_edge("researcher_done", "supervisor")
        builder.add_edge("notes_node", "notes_done")
        builder.add_edge("notes_done", "supervisor")

        builder.add_edge("combine_research", END)

        self.graph = builder.compile(checkpointer=MemorySaver())

        return self.graph
    
    def supervisor_router(self, state: ScholarState) -> str:
        """Route to the tools node if the supervisor makes a tool call."""
        last_message = state["messages"][-1]

        if state["final_answer"]:
            if state["research_data"] and not state["notes_text"] and len(state["research_data"]) > 1:
                return "combine_research"
            return END

        if last_message.tool_calls:
            return "supervisor_tools"
        
        return END
    
    ##############################################################################################
    # HEPLER FUNCTIONS
    ##############################################################################################

    def get_config(self, config_name: str) -> dict:
        """Utility function to load MCP tool configuration from a JSON file."""
        config_path = os.path.join(self.mcp_dir, config_name)
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    ###################################################################################
    # NODE FUNCTIONS
    ###################################################################################

    def load_syllabi(self) -> dict:
        # Go up two levels from src/scholar_agent/ to reach project root
        project_root = os.path.dirname(os.path.dirname(self.base_dir))
        syllabi_path = os.path.join(project_root, "course_syllabus/syllabus.json")
        with open(syllabi_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def syllabus_loader(self, state: ScholarState):
        """Check if the query matches a known course and load its syllabus."""
        query = state["messages"][-1].content if state["messages"] else ""
        syllabi = self.load_syllabi()

        courses = syllabi["courses"]

        llm = self.llm_factory.get_llm_with_structured_output(
            schema=CourseMatchSchema,
            tier=ModelTier.REMOTE
        )

        prompt = self.prompt_manager.get(
            "syllabus_agent", 
            query = query, 
            courses = json.dumps([{"key": k, "course": v["course"]} for k, v in courses.items()], ensure_ascii=False)
        )

        print(prompt)

        try:
            response = await llm.ainvoke([SystemMessage(content=prompt)])
            key = response["matched_key"]
            if key and key in courses:
                matched = courses[key]
                print(f"\n--- SYLLABUS LOADER: matched '{matched['course']}' ---\n")
                return {
                    "course_context": matched,
                    "research_mode": "course_guided"
                }
        except Exception as e:
            print(f"Syllabus loader error: {e}")

        return {
            "course_context": None,
            "research_mode": "general"
        }

    async def planner(self, state: ScholarState):
        """The planner agent, which breaks down the task into smaller sub-tasks."""
        llm = self.llm_factory.get_llm_with_structured_output(
            schema = PlannerTaskSchema, 
            tier = ModelTier.REMOTE
        )

        research_mode = state["research_mode"]
        course_context_str = ""
        if research_mode == "course_guided" and state.get("course_context"):
            course_context_str = json.dumps(state["course_context"], ensure_ascii=False, indent=2)

        prompt = self.prompt_manager.get(
            "planner_agent", 
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            user_request = state["messages"][-1].content if state["messages"] else "None",
            research_mode = state.get("research_mode", "general"),
            course_context = course_context_str
        )

        print("PLANNER")
        print(prompt)

        try:
            response = await llm.ainvoke([SystemMessage(content=prompt)])

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
    
    async def supervisor(self, state: ScholarState):
        """The main supervisor agent."""
        idx = state["current_task_idx"]
        plan = state["plan"] if state["plan"] else []

        if not plan or idx >= len(plan):
            return {
                "final_answer": True,
            }
        
        total = len(plan)
        current_task = plan[idx]

        prompt = self.prompt_manager.get(
            "supervisor_agent",
            idx = idx + 1,
            total = total,
            task_name = current_task["name"],
            task_description = current_task["description"]
        )

        messages = [SystemMessage(content=prompt)]

        try:
            response = await self.llm_with_tools.ainvoke(messages)
        except Exception:
            print("Exception: Non-retryable LLM error")
            raise

        print("\n--- SUPERVISOR NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        return {
            "messages": [response],
        }

    ############################### RESEARCHER ####################################
    
    async def researcher_node(self, state: ScholarState):
        """Researches a topic using MCP tools for RAG and Web search."""
        llm = self.llm_factory.get_tool_llm(tier = ModelTier.REMOTE, tools = self.mcp_tools)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self.mcp_tools
        )

        prompt = self.prompt_manager.get(
            "researcher_agent", 
            query = state["task_description"], 
            tools = tools_context, 
            current_datetime = datetime.now().strftime("%Y-%m-%d")
        )
        
        # Build messages
        messages = [
            SystemMessage(content = prompt)
        ] + state["researcher_messages"]

        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            print(f"RESEARCHER: LLM call failed: {e}")
            return {"researcher_messages": []}
        
        print("\n--- RESEARCHER STATE ---\n")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        print("\n-----------------------\n")

        return {
            "researcher_messages": [response]
        }
    
    def researcher_router(self, state: ScholarState) -> str:
        last_msg = state["researcher_messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "researcher_tools"
        
        return END
    
    async def researcher_done(self, state: ScholarState):
        """Collect the final researcher message into research_data, then return to supervisor."""
        last_msg = state["researcher_messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""

        ai_message = AIMessage(
            name = "researcher",
            content = content
        )

        return {
            "messages": [ai_message],
            "research_data": [content],
            "current_task_idx": state["current_task_idx"] + 1,
        }
    
    async def combine_research(self, state: ScholarState):
        """Combines all research_data entries into a single coherent response."""
        print(f"----- COMBINE RESEARCH -----")
        llm = self.llm_factory.get_remote_llm()

        research_data_str = "\n\n---\n\n".join(state["research_data"])

        prompt = f"""You have received research results on multiple topics. 
            Combine them into a single, well-structured response that covers all topics.
            Do not omit any topic. Preserve all important details.

            Research data:
            {research_data_str}
        """

        try:
            response = await llm.ainvoke([SystemMessage(content=prompt)])
        except Exception as e:
            print(f"COMBINE RESEARCH: failed: {e}")
            # Fall back to just joining the data
            response_content = research_data_str
            return {
                "messages": [AIMessage(content=response_content)]
            }

        return {
            "messages": [AIMessage(content=response.content)]
        }
    
    ############################### NOTES GENERATOR ####################################

    async def notes_node(self, state: ScholarState):
        llm = self.llm_factory.get_remote_llm()

        research_data_str = "\n".join(state["research_data"])

        prompt = self.prompt_manager.get(
            "notes_agent", 
            search_query = state["task_description"], 
            research_data = research_data_str
        )

        try:
            response = await llm.ainvoke([SystemMessage(content = prompt)])
        except Exception as e:
            print(f"NOTES GENERATOR: LLM call failed: {e}")
            return {"notes_text": research_data_str}

        return {
            "notes_text": response.content
        }

    async def notes_done(self, state: ScholarState):
        """Joins all written sections and renders them to a PDF."""

        full_text = state["notes_text"]
        file_path = create_pdf(full_text)

        ai_message = AIMessage(content = f"Study script generated successfully.\n Saved to: {file_path}")

        return {
            "messages": [ai_message],
            "current_task_idx": state["current_task_idx"] + 1,
        }

    ############################### STREAM ####################################
    
    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        state = ScholarState(
            messages=[HumanMessage(content=query)],
            final_answer=False,
            plan=[],
            current_task_idx=0,
            research_data=[],
            researcher_messages=[],
            task_description=None,
            notes_text=None,
            course_context=None,
            research_mode="general",
            notes_sections=[]
        )

        config = RunnableConfig(
            recursion_limit=50,
            configurable={
                "thread_id": context_id,
            }
        )

        last_ai_content = ""
        completed_normally = False

        try:
            async for item in self.graph.astream(
                input = state,
                stream_mode="updates",
                config = config
            ):
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

                                description = await llm_describe_tool_call(tc)
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': description,
                                }

                            elif msg.content:
                                last_ai_content = msg.content.strip()
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': last_ai_content,
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
            if completed_normally:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_ai_content,
                }

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())