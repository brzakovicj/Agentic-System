import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_agenda.log", mode="w")
    ]
)

logger = logging.getLogger()
logger.info("Agenda server starting...")  # confirms logging works and marks process start

import uvicorn

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)


from starlette.applications import Starlette

from src.a2a_services.executors.agent_executor import AgendaAgentExecutor
from src.utils.llm_factory import LLMFactory

# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
#
# The Agent Card is the "business card" of this A2A service.
# It's served automatically at /.well-known/agent-card.json
# Any caller fetches this first to discover what the service can do.
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULE_SKILL = AgentSkill(
    id="schedule",
    name="Schedule",
    description=(
        "Reads users exam schedule and provides a response to a query about the schedule."
    ),
    tags=["schedule", "education", "exam"],
    examples=[
        "What is my exam schedule for tomorrow?",
        "When is the next biology exam?",
    ],
)

AGENDA_AGENT_CARD = AgentCard(
    name="Agenda Service",
    description=(
        "A specialised educational scheduling service built with LangGraph. "
        "Reads and interprets users' exam schedules and answers natural language "
        "queries about upcoming exams, dates, and timetable information. "
        "Designed to help students quickly access and manage academic scheduling information. "
        "Framework-agnostic and compatible with any A2A-enabled agent system."
    ),
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding='JSONRPC',
            url='http://127.0.0.1:9002',
        )
    ],
    skills=[SCHEDULE_SKILL],
)

# ─────────────────────────────────────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────────────────────────────────────

def create_agenda_server():
    """Build the A2A Starlette application."""
    request_handler = DefaultRequestHandler(
        agent_executor=AgendaAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=AGENDA_AGENT_CARD
    )

    routes = []
    routes.extend(create_agent_card_routes(AGENDA_AGENT_CARD))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes=routes)

    return app


if __name__ == "__main__":
    logger.info("[Agenda A2A Service] Starting on http://localhost:9002")
    logger.info("[Agenda A2A Service] Agent Card: "
          "http://localhost:9002/.well-known/agent-card.json")
    logger.info("[Agenda A2A Service] Press Ctrl+C to stop\n")
    LLMFactory.initialize()
    uvicorn.run(create_agenda_server(), host="127.0.0.1", port=9002)