import json
import logging
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
 
from src.host_agent.models import ChatRequest, ChatResponse
from src.host_agent.service import HostAgentService
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    host_service = HostAgentService()
    app.state.host_service = host_service
    logger.info("HostAgentService initialized.")
    yield
    await host_service.client.close()
    logger.info("HostAgentService shut down.")
 
 
app = FastAPI(title="Study Buddy API", lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
 
 
@app.get("/health")
async def health():
    return {"status": "ok"}
 
 
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    host_service = app.state.host_service
    response = await host_service.process_message(request.message)
    return ChatResponse(response=response)
 
 
@app.post("/chat-stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    host_service = app.state.host_service
 
    async def event_generator():
        try:
            async for event in host_service.process_message_stream(request.message, request.context_id):
                # Ako klijent prekine konekciju, prestani da generisujeÅ¡
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, stopping stream.")
                    break

                metadata = event.get("metadata", {}) or {}
 
                yield {
                    "event": event["type"],
                    "data": json.dumps({
                        "content": event["content"],
                        "context_id": event.get("context_id"),  # pass back to client
                        "call_type":  metadata.get("call_type", None),
                        "node_id": metadata.get("node_id", None),
                        "node_name": metadata.get("node_name", None),
                        "node_status": metadata.get("node_status", None),
                        "parent_id": metadata.get("parent_id", None),
                    }),
                }
 
        except Exception:
            logger.exception("Unhandled error in chat-stream")
            yield {
                "event": "error",
                "data": json.dumps({"content": "An unexpected error occurred. Please try again."}),
            }
 
    return EventSourceResponse(event_generator())