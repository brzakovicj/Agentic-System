import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_documents.log", mode="w")
    ]
)

logger = logging.getLogger()
logger.info("Documents server starting...")  # confirms logging works and marks process start

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

from src.a2a_services.executors.documents_executor import DocumentsAgentExecutor
from src.utils.llm_factory import LLMFactory

# ─────────────────────────────────────────────────────────────────────────────
# Agent Card
#
# The Agent Card is the "business card" of this A2A service.
# It's served automatically at /.well-known/agent-card.json
# Any caller fetches this first to discover what the service can do.
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS_SKILL = AgentSkill(
    id="document_management",
    name="Document Management",
    description=(
        "Manages documents stored in a vector database by listing ingested files, "
        "ingesting new documents, rebuilding the database, and safely deleting "
        "documents by verified file names."
    ),
    tags=[
        "documents",
        "vector-db",
        "rag",
        "knowledge-base",
        "ingestion",
        "file-management",
        "retrieval"
    ],
    examples=[
        "List all ingested documents.",
        "What files are currently in the database?",
        "Ingest new documents from the data directory.",
        "Rebuild the vector database from scratch.",
        "Delete notes.pdf from the database.",
        "Remove all lecture files from the vector store.",
    ],
)

DOCUMENTS_AGENT_CARD = AgentCard(
    name="Documents Service",
    description=(
        "A specialized document-management service built for vector database and "
        "RAG knowledge-base operations. The agent strictly operates through "
        "available MCP tools and is responsible for safe, accurate, and minimal "
        "tool execution. Supported operations include listing ingested documents, "
        "ingesting new files, rebuilding the vector database, and deleting "
        "documents by verified file names. The agent prioritizes correctness, "
        "safe deletion workflows, and deterministic tool usage over conversational behavior. "
        "Framework-agnostic and compatible with any A2A-enabled agent system."
    ),
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(
        streaming=True
    ),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="http://127.0.0.1:9004",
        )
    ],
    skills=[DOCUMENTS_SKILL],
)

# ─────────────────────────────────────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────────────────────────────────────

def create_documents_server():
    """Build the A2A Starlette application."""
    request_handler = DefaultRequestHandler(
        agent_executor=DocumentsAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=DOCUMENTS_AGENT_CARD
    )

    routes = []
    routes.extend(create_agent_card_routes(DOCUMENTS_AGENT_CARD))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes=routes)

    return app


if __name__ == "__main__":
    print("[Documents A2A Service] Starting on http://localhost:9004")
    print("[Documents A2A Service] Agent Card: "
          "http://localhost:9002/.well-known/agent-card.json")
    print("[Documents A2A Service] Press Ctrl+C to stop\n")
    LLMFactory.initialize()
    uvicorn.run(create_documents_server(), host="127.0.0.1", port=9004)