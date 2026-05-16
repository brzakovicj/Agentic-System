import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.scholar_agent.scholar.graph import ScholarAgent
from src.scholar_agent.scholar.state import ScholarState

# Ensure src/ is on path when running as script
# sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    TaskState,
    Part
)

from a2a.types.a2a_pb2 import ROLE_AGENT, TASK_STATE_INPUT_REQUIRED, TASK_STATE_WORKING

from a2a.helpers import (
    new_task_from_user_message
)

from a2a.server.tasks import TaskUpdater

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

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TASK_STATE_WORKING,
                        message = Message(
                            message_id=str(uuid4()),
                            role=ROLE_AGENT,
                            parts=[Part(text=item['content'])],
                            context_id=task.context_id,
                            task_id=task.id,
                        )
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

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Raise exception as cancel is not supported."""
        raise Exception('cancel not supported')
    