import json
import logging
import os
from datetime import datetime
from typing import Any, AsyncIterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RunnableConfig, interrupt

from src.agenda_agent.state import AgendaState

from src.utils.tool_formatter import llm_describe_tool_call
from src.utils.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory, ModelTier
from src.utils.mcp_client import MCPClient
from src.utils.url_cache import load_cached_url, save_cached_url

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgendaAgent:
    def __init__(self) -> None:
        self._graph: Any = None
        self._prompt_manager = PromptManager()
        self._llm_factory = LLMFactory.get_instance()

        _base = os.path.dirname(os.path.abspath(__file__))
        self._mcp_dir = os.path.join(_base, "mcp")

        self._agenda_config = self._load_config("config.json")
        self.mcp_client = MCPClient(self._agenda_config)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self, filename: str) -> dict:
        config_path = os.path.join(self._mcp_dir, filename)
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        return config

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    async def build_graph(self) -> Any:
        self._agenda_tools = await self.mcp_client.get_tools()

        builder = StateGraph(AgendaState)
        
        # agenda
        builder.add_node("agenda_init", self.agenda_init)
        builder.add_node("agenda_node", self.agenda_node)
        builder.add_node("agenda_tools", ToolNode(self._agenda_tools))

        # Entry
        builder.set_entry_point("agenda_init")

        builder.add_edge("agenda_init", "agenda_node")

        builder.add_conditional_edges(
            "agenda_node",
            self._agenda_router,
            {
                "agenda_tools": "agenda_tools",
                END: END
            },
        )

        builder.add_edge("agenda_tools", "agenda_node")

        # --------------------- END --------------------------

        self._graph = builder.compile(checkpointer=MemorySaver())
        
        return self._graph

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def _agenda_router(self, state: AgendaState) -> str:
        last = state["messages"][-1]

        # Still has pending tool calls → continue the tool loop
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "agenda_tools"

        return END

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    
    async def agenda_init(self, state: AgendaState, config: RunnableConfig):
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
            logger.exception("agenda_init_node: LLM call failed")
            response = "NONE"

        url: str | None = response if response != "NONE" else None
        
        # 1. Fall back to persisted state
        if not url:
            url = state.get("agenda_url", None)
            logger.info("STATE URL %s", url)

        # 2. Fall back to cached URL for this thread
        if not url:
            url = load_cached_url()
            if url:
                logger.info("CACHE URL %s", url)
        
        # 3. Fall back interrupt
        if not url:
            url = interrupt({
                "message": (
                    "Please send the link to the exam schedule "
                    "(e.g. https://university.edu/raspored.pdf)."
                ),
                "type": "text_input",
                "placeholder": "https://university.edu/raspored.pdf",
            })

            logger.info("INTERRUPT URL %s", url)

        save_cached_url(url)

        return {
            "messages": [AIMessage(name="agenda_init", content="Exam schedule URL initialized.")],
            "agenda_url": url,
        }

    async def agenda_node(self, state: AgendaState) -> dict:
        """
        Call the LLM with the exam-schedule URL and available MCP tools.
        Saves the final text response as ``agenda_summary``.
        """
        llm = self._llm_factory.get_tool_llm(
            tier=ModelTier.REMOTE,
            tools=self._agenda_tools,
        )

        tools_ctx = "\n".join(
            f"{t.name}: {t.description}" 
            for t in self._agenda_tools
        )

        print(f"TOOL context: {tools_ctx}" )

        system_prompt = self._prompt_manager.get(
            "agenda_agent",
            tool_context=tools_ctx,
            url=state["agenda_url"] or "",
            user_query=state["user_input"] or "",
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            response = await llm.ainvoke(messages)
        except Exception:
            logger.exception("agenda_node: LLM call failed")
            raise

        print("\n--- AGENDA NODE ---")
        print(type(response).__name__, getattr(response, "content", ""))
        if hasattr(response, "tool_calls"):
            print("TOOL CALLS:", response.tool_calls)

        return {
            "messages": [response]
        }

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        config = RunnableConfig(
            recursion_limit=50,
            configurable={
                "thread_id": context_id,
            }
        )

        logger.info("THREAD ID %s", context_id)

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
            state = AgendaState(
                messages=[HumanMessage(content=query)],
                user_input=query
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
                            "call_type": None,
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

                                    description = await llm_describe_tool_call(tc)
                                    yield {
                                        'is_task_complete': False,
                                        'require_user_input': False,
                                        'content': description,
                                        "call_type": "tool",
                                    }

                                elif msg.content:
                                    last_ai_content = msg.content.strip()
                                    yield {
                                        'is_task_complete': False,
                                        'require_user_input': False,
                                        'content': last_ai_content,
                                        "call_type": None,
                                    }

            completed_normally = True

        except Exception as exc:
            print(f"Graph execution failed: {exc}")

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(exc)}",
                "call_type": None,
            }

        finally:
            if completed_normally and not interrupted:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_ai_content,
                    "call_type": None,
                }