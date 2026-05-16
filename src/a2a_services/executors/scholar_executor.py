from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers import new_task_from_user_message
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    Part,
    Role,
    TaskState,
)
from uuid_utils import uuid4

from src.scholar_agent.scholar.graph import ScholarAgent

class ScholarAgentExecutor(AgentExecutor):
    """
    A2A executor for ScholarAgent.
 
    The graph is built lazily on the first call so the constructor stays
    synchronous (A2A may instantiate executors without an event loop).
    """

    def __init__(self) -> None: 
        self._agent = None  # built lazily

    async def _build_graph(self):
        """Return the compiled graph, building it on first call."""
        if self._agent is None:
            self._agent = ScholarAgent()
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

                agent_message = Message(
                    message_id=str(uuid4()),
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=item['content'])],
                    context_id=task.context_id,
                    task_id=task.id,
                )

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=agent_message,
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.TASK_STATE_INPUT_REQUIRED,
                        message=agent_message,
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

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Raise exception as cancel is not supported."""
        raise Exception('cancel not supported')
    