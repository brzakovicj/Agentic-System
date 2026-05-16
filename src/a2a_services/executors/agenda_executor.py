import sys
from pathlib import Path

from langchain.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from uuid_utils import uuid4

from src.agenda_agent.agenda.graph import AgendaAgent
from src.agenda_agent.agenda.state import AgendaState

# Ensure src/ is on path when running as script
#sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    Part,
    TaskState
)

from a2a.types.a2a_pb2 import ROLE_AGENT, TASK_STATE_INPUT_REQUIRED, TASK_STATE_WORKING

from a2a.helpers import (
    new_task_from_user_message
)

from a2a.server.tasks import TaskUpdater

# def _extract_user_text(context: RequestContext) -> str:
#     """
#     Pull the plain-text content out of an A2A RequestContext.
 
#     A2A messages carry a list of Parts; we grab the first text part.
#     """
#     for part in context.message.parts or []:
#         # SDK uses a union type – the text variant has a `.text` attribute
#         if hasattr(part, "text") and part.text:
#             return part.text
#         # Some SDK versions wrap it in a `.root` discriminated union
#         if hasattr(part, "root") and hasattr(part.root, "text"):
#             return part.root.text
#     return ""
 
 
# def _collect_final_response(graph_state: dict) -> str:
#     """
#     Walk the final graph state and return the last non-empty AI message content.
#     Falls back to a generic string if nothing useful is found.
#     """
#     messages = graph_state.get("messages", [])
#     for msg in reversed(messages):
#         if isinstance(msg, AIMessage) and msg.content:
#             return msg.content
#     return "Agent completed with no textual output."
 

class AgendaAgentExecutor(AgentExecutor):
    """
    Handles incoming A2A agenda tasks.

    Request format (JSON in the text part):
    {
        "query":       "What are the key developments in renewable energy technology?",
    }

    Response format (JSON in the text part):
    {
        "response":   "Renewable energy technology has seen significant advancements in recent years, including improvements in solar panel efficiency, the development of more efficient wind turbines, and breakthroughs in energy storage solutions such as solid-state batteries. Additionally, there has been progress in green hydrogen production and carbon capture technologies, all contributing to a more sustainable energy future.",
    }
    """

    def __init__(self) -> None:
        self._agent = None

    async def _build_graph(self):
        if self._agent is None:
            self._agent = AgendaAgent()
            await self._agent.build_graph()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent process and enqueue the final response."""
        error = self._validate_request(context)
        if error:
            raise InvalidParamsError()
        
        query = context.get_user_input()
        task = context.current_task

        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            await self._build_graph()
            context_id = context.context_id or context.task_id or "a2a-default"

            async for item in self._agent.astream(
                query = query, 
                context_id = context_id
            ):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TASK_STATE_WORKING,
                        message = Message(
                            message_id=str(uuid4()),
                            role=ROLE_AGENT,
                            parts=[Part(text=item['content'])],
                            context_id=task.context_id,
                            task_id=task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TASK_STATE_INPUT_REQUIRED,
                        message = Message(
                            message_id=str(uuid4()),
                            role=ROLE_AGENT,
                            parts=[Part(text=item['content'])],
                            context_id=task.context_id,
                            task_id=task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(text=item['content'])],
                        name='conversion_result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            raise InternalError() from e


        # await event_queue.enqueue_event(
        #     TaskStatusUpdateEvent(
        #         task_id = context.task_id,
        #         context_id = context.context_id,
        #         status = TaskStatus(
        #             state = TaskState.TASK_STATE_WORKING,
        #             message = new_text_message('Processing request...'),
        #         ),
        #     )
        # )

        # user_text = _extract_user_text(context)
        # if not user_text:
        #     user_text = "No query provided."
 
        # graph_input = self._state(
        #     messages=[HumanMessage(content=user_text)],
        #     final_answer=False,
        # )
 
        # thread_id = context.context_id or context.task_id or "a2a-default"
        # config = RunnableConfig(configurable={
        #     "thread_id": thread_id,
        #     "recursion_limit": 50,
        # })
 
        # graph = await self._get_graph()
        # final_state = await graph.ainvoke(input=graph_input, config=config)
        # result_text = _collect_final_response(final_state)

        # await event_queue.enqueue_event(
        #     TaskArtifactUpdateEvent(
        #         task_id = context.task_id,
        #         context_id = context.context_id,
        #         artifact = new_text_artifact(name='result', text=result_text),
        #     )
        # )
        # await event_queue.enqueue_event(
        #     TaskStatusUpdateEvent(
        #         task_id = context.task_id,
        #         context_id = context.context_id,
        #         status = TaskStatus(
        #             state = TaskState.TASK_STATE_COMPLETED
        #         ),
        #     )
        # )

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Raise exception as cancel is not supported."""
        raise Exception('cancel not supported')