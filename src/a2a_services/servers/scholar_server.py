import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_scholar.log", mode="w")
    ]
)

logger = logging.getLogger()
logger.info("Scholar server starting...")  # confirms logging works and marks process start

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

from src.a2a_services.executors.agent_executor import ScholarAgentExecutor
from src.utils.llm_factory import LLMFactory

# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
#
# The Agent Card is the "business card" of this A2A service.
# It's served automatically at /.well-known/agent-card.json
# Any caller fetches this first to discover what the service can do.
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_SKILL = AgentSkill(
    id="research",
    name="Research",
    description=(
        "Researches a topic using users course materials andweb search and returns a concise summary of the key points. "
    ),
    tags=["research", "rag", "education", "web search"],
    examples=[
        "Tell me something about Olympic history.",
        "What are the key developments in renewable energy technology?",
    ],
)

NOTES_GENERATION_SKILL = AgentSkill(
    id="notes_generation",
    name="Notes Generation",
    description=(
        "Generates concise notes on a given topic based on users course materials and web search."
    ),
    tags=["notes", "script", "summary", "PDF", "research"],
    examples=[
        "Generate notes on the history of the Olympic Games.",
        "Create concise notes on the latest advancements in renewable energy technology.",
    ],
)

SCHOLAR_AGENT_CARD = AgentCard(
    name="Scholar Service",
    description=(
        "A specialised educational assistant service built with LangGraph. "
        "Provides two core capabilities: researching topics using users' course "
        "materials together with web search, and generating concise structured "
        "notes based on the gathered information. "
        "Designed for learning support, summarisation, and knowledge synthesis. "
        "Framework-agnostic and compatible with any A2A-enabled agent system."
    ),
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding='JSONRPC',
            url='http://127.0.0.1:9001',
        )
    ],
    skills=[RESEARCH_SKILL, NOTES_GENERATION_SKILL],
)

# ─────────────────────────────────────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────────────────────────────────────

def create_scholar_server():
    """Build the A2A Starlette application."""
    request_handler = DefaultRequestHandler(
        agent_executor=ScholarAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=SCHOLAR_AGENT_CARD
    )

    routes = []
    routes.extend(create_agent_card_routes(SCHOLAR_AGENT_CARD))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes=routes)

    return app


if __name__ == "__main__":
    print("[Scholar A2A Service] Starting on http://localhost:9001")
    print("[Scholar A2A Service] Agent Card: "
          "http://localhost:9001/.well-known/agent-card.json")
    print("[Scholar A2A Service] Press Ctrl+C to stop\n")
    LLMFactory.initialize()
    uvicorn.run(create_scholar_server(), host="127.0.0.1", port=9001)
