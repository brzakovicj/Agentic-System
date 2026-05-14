import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.researcher_agent.supervisor.graph import SupervisorAgent

# Ensure src/ is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from a2a.helpers import (
    new_task_from_user_message,
    new_text_artifact,
    new_text_message,
)

class ResearcherAgentExecutor(AgentExecutor):
    """
    Handles incoming A2A research tasks.

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
        self.agent = SupervisorAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent process and enqueue the final response."""
        task = context.current_task or new_task_from_user_message(
            context.message
        )
        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                status = TaskStatus(
                    state = TaskState.TASK_STATE_WORKING,
                    message = new_text_message('Processing request...'),
                ),
            )
        )

        result = await self.agent.invoke()

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                artifact = new_text_artifact(name='result', text=result),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id = context.task_id,
                context_id = context.context_id,
                status = TaskStatus(
                    state = TaskState.TASK_STATE_COMPLETED
                ),
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Raise exception as cancel is not supported."""
        raise Exception('cancel not supported')
    