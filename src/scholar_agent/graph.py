import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, AsyncIterable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RunnableConfig
from src.scholar_agent.state import ScholarState
from src.scholar_agent.tools import PlannerTaskSchema, create_pdf, handoff_to_subagent
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient

from tenacity import RetryError

from src.utils.retryable_invoke import ainvoke_llm

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

        builder.add_node("planner", self.planner)
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("supervisor_tools", ToolNode(self.tools))

        # Researcher
        builder.add_node("researcher_node", self.researcher_node)
        builder.add_node("researcher_tools", ToolNode(self.mcp_tools))
        builder.add_node("researcher_done", self.researcher_done)
        
        # Notes generator
        builder.add_node("notes_node", self.notes_node)
        builder.add_node("notes_done", self.notes_done)

        # ----- EDGES -----

        builder.set_entry_point("planner")
        builder.add_edge("planner", "supervisor")

        builder.add_conditional_edges(
            "supervisor",
            self.supervisor_router,
            {
                "tools": "supervisor_tools",
                END: END,
            }
        )

        builder.add_conditional_edges(
            "researcher_node",
            self.researcher_router,
            {
                "tools": "researcher_tools",
                "end": "researcher_done",
            }
        )

        builder.add_edge("researcher_tools", "researcher_node")
        builder.add_edge("researcher_done", "supervisor")
        builder.add_edge("notes_node", "notes_done")
        builder.add_edge("notes_done", "supervisor")

        self.graph = builder.compile(checkpointer=MemorySaver())

        return self.graph
    
    def supervisor_router(self, state: ScholarState) -> str:
        """Route to the tools node if the supervisor makes a tool call."""
        last_message = state["messages"][-1]

        if state["final_answer"]:
            return END

        if last_message.tool_calls:
            return "tools"
        
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

    async def planner(self, state: ScholarState):
        """The planner agent, which breaks down the task into smaller sub-tasks."""
        llm = self.llm_factory.get_llm_with_structured_output(
            schema = PlannerTaskSchema, 
            tier = ModelTier.REMOTE
        )

        prompt = self.prompt_manager.get(
            "task_planner_prompt", 
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            user_request = state["messages"][-1].content if state["messages"] else "None")

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

        # Provide only the two most recent messages as context to avoid
        # inflating the prompt with the entire messages on every iteration.
        recent = state["messages"][-2:] if state["messages"] else []
        messages = "\n\n".join(
            m.content for m in recent
            if hasattr(m, "content") and m.content
        )

        prompt = self.prompt_manager.get(
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

    ############################### RESEARCHER ####################################
    
    async def researcher_node(self, state: ScholarState):
        """Researches a topic using MCP tools for RAG and Web search."""
        llm = self.llm_factory.get_tool_llm(tier = ModelTier.REMOTE, tools = self.mcp_tools)
        
        tools_context = "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self.mcp_tools
        )

        prompt = self.prompt_manager.get(
            "researcher_single_agent_prompt", 
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
    
    async def researcher_router(self, state: ScholarState) -> str:
        last_msg = state["researcher_messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        
        return "end"
    
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
        }
    
    ############################### NOTES GENERATOR ####################################

    async def notes_node(self, state: ScholarState):
        llm = self.llm_factory.get_remote_llm()

        research_data_str = "\n".join(state["research_data"])

        prompt = self.prompt_manager.get(
            "notes_single_agent_prompt", 
            search_query = state["task_description"], 
            research_data = research_data_str
        )

        try:
            response = await llm.ainvoke([SystemMessage(content = prompt)])
        except Exception as e:
            print(f"NOTES GENERATOR: LLM call failed: {e}")
            return {"notes_text": []}

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
        }

    ############################### STREAM ####################################
    
    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        state = ScholarState(
            messages = [ HumanMessage(content = query) ],
            final_answer = False
        )

        config = RunnableConfig(configurable={
            "thread_id": context_id,
            "recursion_limit": 50,
        })

        last_ai_content = ""

        async for item in self.graph.astream(
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
                            last_ai_content = msg.content.strip()
                            yield {
                                'is_task_complete': True,
                                'require_user_input': False,
                                'content': msg.content.strip(),
                            }

                        elif msg.content:
                            last_ai_content = msg.content.strip()
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
                        'content': last_ai_content,
                    }

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())