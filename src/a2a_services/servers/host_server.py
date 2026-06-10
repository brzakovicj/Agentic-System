from fastapi.middleware import Middleware
import uvicorn

from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore

from src.a2a_services.executors.agent_executor import HostAgentExecutor

from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentSkill,
)

from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):

        if request.method == "POST":
            body = await request.body()

            print("\n=== JSON RPC REQUEST ===")
            print(body.decode())
            print("========================\n")

        return await call_next(request)

HOST_AGENT_CARD = AgentCard(
    name="Host Agent",
    description=(
        "Main orchestration agent that routes requests "
        "to specialised educational agents."
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
            url="http://127.0.0.1:9005",
        )
    ],
    skills=[
        AgentSkill(
            id="routing",
            name="Request Routing",
            description=(
                "Routes requests to Scholar, Agenda, Study Plan "
                "and Documents agents."
            ),
            tags=["routing", "orchestration"],
            examples=[
                "Generate notes about databases",
                "Create a study plan",
                "When is my next exam?"
            ],
        )
    ],
)

def create_host_server():

    request_handler = DefaultRequestHandler(
        agent_executor=HostAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=HOST_AGENT_CARD,
    )

    routes = []
    routes.extend(
        create_agent_card_routes(HOST_AGENT_CARD)
    )
    routes.extend(
        create_jsonrpc_routes(
            request_handler,
            "/",
            enable_v0_3_compat=True
        )
    )

    return Starlette(routes=routes, middleware=[
        Middleware(RequestLoggerMiddleware)
    ])


if __name__ == "__main__":
    uvicorn.run(
        create_host_server(),
        host="127.0.0.1",
        port=9005,
    )