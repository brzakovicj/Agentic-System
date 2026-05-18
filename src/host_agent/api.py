from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.host_agent.models import (
    ChatRequest,
    ChatResponse
)
from src.host_agent.service import HostAgentService
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):

    # startup
    host_service = HostAgentService()

    app.state.host_service = host_service

    yield

    # shutdown
    await host_service.client.close()

app = FastAPI(lifespan=lifespan)

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# host_service = HostAgentService()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    host_service = app.state.host_service

    response = await host_service.process_message(
        request.message
    )

    return ChatResponse(
        response=response
    )