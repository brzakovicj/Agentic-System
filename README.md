# 📚 Study Assistant — Multi-Agent AI System

A multi-agent study assistant built on an **A2A (Agent-to-Agent)** architecture with **MCP (Model Context Protocol)** servers. It routes user queries through a host service to specialized agents that handle research, scheduling, document management, and personalized study planning.

---

## Architecture Overview

```
User → UI → Router/Host Agent → A2A Agent Servers → MCP Servers → Data Sources
```

The system is composed of four layers:

| Layer | Components |
|---|---|
| **Router** | Host Service — receives queries and dispatches to the right agent |
| **A2A Agent Servers** | Scholar, Agenda, Study Plan, Documents |
| **MCP Servers** | RAG MCP Server, PDF Reader MCP Server, ChromaDB MCP Server |
| **Data Sources** | ChromaDB, Exam Schedule (URL), Course Syllabus (JSON) |

---

## Features

### 🔍 Research
Ask any question and the Scholar agent gathers comprehensive information via RAG.
> *"Research the causes of World War I"*

### 📝 Study Notes
Generates clean, structured notes on any topic — ready for revision.
> *"Generate study notes on the water cycle"*

### 📅 Exam Schedule
Pulls your upcoming exam schedule from a live URL source.
> *"What exams do I have this week?"*

### 🗓️ Study Plan Generation
Creates personalized study plans based on your exam schedule and course syllabus.
> *"Create a study plan for my upcoming Database 2 exam"*

### 🧠 Topic Q&A
Answers subject-specific questions with depth, powered by your document database.
> *"Explain how mitosis works"*

### 🗂️ Document Database Management
Manage the underlying ChromaDB document store directly.
> *"List all documents in the database"*  
> *"Ingest new files"*  
> *"Delete lecture_notes.pdf"*  
> *"Reset the database"*

---

## Agents

### Host Service (Router/Host Agent)
The central dispatcher. Receives all user queries from the UI and routes them to the appropriate A2A agent based on intent.

### Scholar A2A Server
Handles research and Q&A tasks. Uses the **RAG MCP Server** and **PDF Reader MCP Server** to retrieve relevant information from ingested documents and answer topic-based questions.

### Agenda A2A Server
Manages exam schedule queries. Reads from the **Exam Schedule (URL)** data source and can interact with the **ChromaDB MCP Server** for context.

### Study Plan A2A Server
Generates personalized study plans. Communicates with other agents via A2A and uses the **Course Syllabus (JSON)** and exam schedule to build tailored plans.

### Documents A2A Server
Handles all document database operations — listing, ingesting, deleting, and resetting the ChromaDB store via the **ChromaDB MCP Server**.

---

## MCP Servers

| Server | Responsibility |
|---|---|
| **RAG MCP Server** | Retrieval-augmented generation over ingested documents |
| **PDF Reader MCP Server** | Extracts and processes content from PDF files |
| **ChromaDB MCP Server** | Direct interface to the ChromaDB vector store |

---

## Data Sources

| Source | Type | Used By |
|---|---|---|
| **ChromaDB** | Vector database | RAG, ChromaDB MCP Server |
| **Exam Schedule** | Live URL | Agenda agent |
| **Course Syllabus** | JSON file | Scholar, Study Plan agent |

---

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
git clone https://github.com/your-org/study-assistant.git
cd study-assistant
uv sync        # or: pip install -e .
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Refer to `.env.example` in the root for all required variables (API keys, MCP server config, etc.).

### Running the System

Each component runs as a separate process. Start them **in this order**, each in its own terminal.

**1. Activate the virtual environment** (do this in every terminal before running any command):

```bash
.venv\Scripts\activate
```

**2. Start ChromaDB:**

```bash
chroma run --host localhost --port 8000
```

**3. Start the A2A agent servers:**

```bash
uv run -m src.a2a_services.servers.scholar_server
uv run -m src.a2a_services.servers.agenda_server
uv run -m src.a2a_services.servers.study_plan_server
uv run -m src.a2a_services.servers.documents_server
```

**4. Start the host agent API:**

```bash
uv run uvicorn src.host_agent.api:app --reload --port 8001
```

**5. Launch the UI:**

```bash
uv run streamlit run src/ui/streamlit_app.py
```

Then open the Streamlit URL in your browser and start asking questions.

---

## Project Structure

```
study-assistant/
├── course_syllabus/
│   └── syllabus.json               # Course syllabus data used by the Study Plan agent
├── outputs/                        # Generated outputs (study plans, notes, etc.)
├── study_materials/                # Source documents for ingestion into ChromaDB
├── src/
│   ├── a2a_services/
│   │   ├── executors/
│   │   │   ├── agent_executor.py   # Runs agent graphs in response to A2A requests
│   │   │   └── base_agent_executor.py
│   │   └── servers/
│   │       ├── agenda_server.py
│   │       ├── documents_server.py
│   │       ├── scholar_server.py
│   │       └── study_plan_server.py
│   ├── agenda_agent/               # Exam schedule agent (LangGraph)
│   │   ├── mcp/config.json
│   │   ├── graph.py
│   │   └── state.py
│   ├── documents_agent/            # Document DB management agent (LangGraph)
│   │   ├── mcp/config.json
│   │   ├── graph.py
│   │   └── state.py
│   ├── host_agent/                 # Router — receives queries and dispatches to agents
│   │   ├── api.py                  # FastAPI entrypoint
│   │   ├── models.py
│   │   └── service.py
│   ├── mcp_servers/
│   │   ├── chromadb_server/        # ChromaDB MCP server + SQLite file registry
│   │   └── rag_server/             # RAG MCP server
│   ├── prompts/                    # Markdown prompt files for each agent/sub-agent
│   │   ├── agenda_agent.md
│   │   ├── documents_agent.md
│   │   ├── host_agent.md
│   │   ├── notes_agent.md
│   │   ├── planner_agent.md
│   │   ├── researcher_agent.md
│   │   ├── study_plan_agent.md
│   │   ├── supervisor_agent.md
│   │   ├── syllabus_agent.md
│   │   └── prompt_manager.py
│   ├── scholar_agent/              # Research & Q&A agent (LangGraph)
│   │   ├── mcp/config.json
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── study_plan_agent/           # Study plan generation agent (LangGraph)
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── ui/
│   │   └── streamlit_app.py        # Streamlit chat interface
│   └── utils/
│       ├── llm_factory.py          # LLM client factory
│       ├── mcp_client.py           # Shared MCP client helper
│       └── retryable_invoke.py     # Retry logic for LLM calls
├── pyproject.toml
└── .env.example
```

---

## License
