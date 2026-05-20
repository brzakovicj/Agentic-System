import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_study_plan.log", mode="w")
    ]
)

logger = logging.getLogger()
logger.info("Study plan server starting...")  # confirms logging works and marks process start

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
from src.a2a_services.executors.study_plan_executor import StudyPlanAgentExecutor
from src.utils.llm_factory import LLMFactory

# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
#
# The Agent Card is the "business card" of this A2A service.
# It's served automatically at /.well-known/agent-card.json
# Any caller fetches this first to discover what the service can do.
# ─────────────────────────────────────────────────────────────────────────────

STUDY_PLAN_GENERATION_SKILL = AgentSkill(
    id="study_plan_generation",
    name="Study Plan Generation",
    description=(
        "Builds a structured, time-aware study plan based on the exam date and "
        "material scope. Supports daily breakdowns for short timelines (≤14 days) "
        "and weekly breakdowns for longer ones. Always reserves the last 1/2 days "
        "for revision. Generates an emergency condensed review plan if fewer than "
        "3 days remain before the exam."
    ),
    tags=["study plan", "schedule", "education", "planning", "exam prep"],
    examples=[
        "Create a study plan for my biology exam on 2025-06-15.",
        "I have a math exam in 3 weeks, help me plan my preparation.",
        "My exam is tomorrow, what should I focus on?",
    ],
)

STUDY_PLAN_AGENT_CARD = AgentCard(
    name="Study Plan Service",
    description=(
        "Builds a structured, time-aware study plan based on the exam date and "
        "material scope. Produces daily or weekly task breakdowns based on available time, "
        "reserves revision days, and handles tight deadlines with emergency plans. "
        "Compatible with any A2A-enabled agent system."
    ),
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="http://127.0.0.1:9003",
        )
    ],
    skills=[
        STUDY_PLAN_GENERATION_SKILL,
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────────────────────────────────────

def create_study_plan_server():
    """Build the A2A Starlette application."""
    request_handler = DefaultRequestHandler(
        agent_executor=StudyPlanAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=STUDY_PLAN_AGENT_CARD
    )

    routes = []
    routes.extend(create_agent_card_routes(STUDY_PLAN_AGENT_CARD))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes=routes)

    return app


if __name__ == "__main__":
    print("[Study Plan A2A Service] Starting on http://localhost:9003")
    print("[Study Plan A2A Service] Agent Card: "
          "http://localhost:9003/.well-known/agent-card.json")
    print("[Study Plan A2A Service] Press Ctrl+C to stop\n")
    LLMFactory.initialize()
    uvicorn.run(create_study_plan_server(), host="127.0.0.1", port=9003)
