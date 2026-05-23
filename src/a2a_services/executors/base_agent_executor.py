from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Generic, TypeVar
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

from src.agenda_agent.graph import AgendaAgent

T = TypeVar("T")

logger = logging.getLogger(__name__)

class BaseAgentExecutor(AgentExecutor, Generic[T], ABC):
    """
    Handles incoming A2A tasks.

    Thread-safe executor that manages agent lifecycle and task state.
    Designed for concurrent use across multiple simultaneous requests.
    """

    def __init__(self) -> None:
        self._running_tasks: set[str] = set()
        self._agent: T | None = None

        # Prevents race condition during lazy agent initialization
        self._agent_init_lock = asyncio.Lock()

        # Protects _running_tasks from concurrent read/write
        self._tasks_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses define these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _create_agent(self) -> T:
        """
        Instantiate and initialize the agent.

        Example:
            agent = MyAgent()
            await agent.build_graph()
            return agent
        """
    
    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def _get_agent(self) -> T:
        if self._agent is not None:
            return self._agent

        async with self._agent_init_lock:
            if self._agent is None:
                logger.info("[%s] Initializing agent...", self.__class__.__name__)
                self._agent = await self._create_agent()
                logger.info("[%s] Agent initialized.", self.__class__.__name__)

        return self._agent
    
    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    async def _track_task(self, task_id: str) -> None:
        async with self._tasks_lock:
            self._running_tasks.add(task_id)

    async def _untrack_task(self, task_id: str) -> None:
        async with self._tasks_lock:
            self._running_tasks.discard(task_id)
        
    # ------------------------------------------------------------------
    # Core execution — shared across all agents
    # ------------------------------------------------------------------

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent process and enqueue the final response."""
        error = self._validate_request(context)
        if error:
            raise InvalidParamsError()
        
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_id = task.id
        context_id = context.context_id or task_id

        updater = TaskUpdater(event_queue, task_id, task.context_id)
        await self._track_task(task_id)

        try:
            agent = await self._get_agent()

            async for item in agent.astream(
                query = context.get_user_input(), 
                context_id = context_id
            ):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                content = item["content"]

                agent_message = Message(
                    message_id=str(uuid4()),
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=content)],
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
                    logger.info("[%s] Task %s waiting for input.", self.__class__.__name__, task_id)
                    break
                else:
                    await updater.add_artifact(
                        [Part(text=content)],
                        name='conversion_result',
                    )
                    await updater.complete()
                    logger.info("[%s] Task %s completed.", self.__class__.__name__, task_id)
                    break

        except Exception as e:
            logger.exception("[%s] Unhandled error in task %s", self.__class__.__name__, task_id)
            raise InternalError() from e
        finally:
            await self._untrack_task(task_id)

    def _validate_request(self, context: RequestContext) -> bool:
        if not context.message:
            return True
        if not context.get_user_input():
            return True
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancels a task."""
        task = context.current_task
        if task is None:
            logger.warning("[%s] Cancel called with no current task.", self.__class__.__name__)
            return

        await self._untrack_task(task.id)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=context.context_id or "",
        )
        await updater.cancel()
        logger.info("[%s] Task %s cancelled.", self.__class__.__name__, task.id)