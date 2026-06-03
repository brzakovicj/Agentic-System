# Study Assistant

A multi-agent AI system that routes your study-related queries to specialized agents for research, scheduling, document management, and personalized study planning.

---

## Architecture

```
User → UI → Host Agent → A2A Agent Servers → MCP Servers → Data Sources
```

| Layer            | Components                                                         |
| ---------------- | ------------------------------------------------------------------ |
| **Router**       | Host Agent — receives queries and dispatches to the right agent    |
| **A2A Agents**   | Scholar, Agenda, Study Plan, Documents                             |
| **MCP Servers**  | RAG, PDF Reader, ChromaDB, DuckDuckGo                              |
| **Data Sources** | ChromaDB vector store, Exam Schedule (URL), Course Syllabus (JSON) |

---

## What You Can Do

| Task                           | Example                                                 |
| ------------------------------ | ------------------------------------------------------- |
| Research any topic             | _"Research the causes of World War I"_                  |
| Generate study notes           | _"Generate study notes on the water cycle"_             |
| Check your exam schedule       | _"What exams do I have this week?"_                     |
| Create a study plan            | _"Create a study plan for my upcoming Database 2 exam"_ |
| Ask subject-specific questions | _"Explain how mitosis works"_                           |
| Manage your document database  | _"List all documents"_ / _"Delete lecture_notes.pdf"_   |

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

```bash
cp .env.example .env
```

Fill in your values — refer to `.env.example` for all required variables (API keys, MCP server config, etc.).

### Running the System

Each component runs as a separate process. **Activate the virtual environment in every terminal before running any command:**

```bash
.venv\Scripts\activate
```

Then start the components **in order**:

**1. ChromaDB**

```bash
chroma run --host localhost --port 8000
```

**2. A2A Agent Servers** (each in its own terminal)

```bash
uv run -m src.a2a_services.servers.scholar_server
uv run -m src.a2a_services.servers.agenda_server
uv run -m src.a2a_services.servers.study_plan_server
uv run -m src.a2a_services.servers.documents_server
```

**3. Host Agent API**

```bash
uv run uvicorn src.host_agent.api:app --reload --port 8001
```

**4. UI**

```bash
uv run streamlit run src/ui/streamlit_app.py
```

Open the Streamlit URL in your browser and start asking questions.

---

## Agents

### Host Agent

The central dispatcher. Analyzes every incoming query and routes it to the appropriate specialized agent via the A2A protocol.

### Scholar Agent

Handles research and Q&A. Uses RAG over your ingested documents and falls back to web search (via DuckDuckGo MCP) when needed. Its internal graph is composed of four nodes:

- **Planner** — breaks the request into tasks
- **Supervisor** — manages task execution
- **Researcher** — collects information
- **Notes Generator** — formats output for PDF generation

Generated study notes are saved to the `outputs/` directory and available for download in the UI. For subject-specific requests (e.g. _"notes for Operating Systems 2"_), the agent loads topic data from `syllabus.json`.

### Agenda Agent

Fetches and answers questions about your exam schedule. Reads from a user-provided URL using the PDF Reader MCP server. The system remembers the last used URL; you can view or clear it from the UI.

### Study Plan Agent

Generates personalized study plans based on your course syllabus and exam dates. If no exam date is provided, it automatically queries the Agenda Agent via A2A to retrieve the information. Subject syllabi are stored locally as JSON files.

### Documents Agent

Manages the ChromaDB knowledge base through the ChromaDB MCP server. Supported operations:

- List documents currently in the database
- Ingest new documents
- Delete existing documents
- Reset the database (re-indexes all files in `study_materials/`)

---

## MCP Servers

| Server                    | Responsibility                                         |
| ------------------------- | ------------------------------------------------------ |
| **RAG MCP Server**        | Retrieval-augmented generation over ingested documents |
| **PDF Reader MCP Server** | Extracts and processes content from PDF files          |
| **ChromaDB MCP Server**   | Direct interface to the ChromaDB vector store          |
| **DuckDuckGo MCP Server** | Provides web search capabilities through DuckDuckGo    |

---

## Project Structure

```
study-assistant/
├── course_syllabus/
│   └── syllabus.json               # Course syllabus data used by Study Plan agent and Scholar agent
├── outputs/                        # Generated outputs (study notes)
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
│   ├── documents_agent/            # Document DB management agent (LangGraph)
│   ├── host_agent/                 # Router — receives queries and dispatches to agents
│   │   ├── api.py                  # FastAPI entrypoint
│   │   ├── models.py
│   │   └── service.py
│   ├── mcp_servers/
│   │   ├── chromadb_server/        # ChromaDB MCP server + SQLite file registry
│   │   └── rag_server/             # RAG MCP server
│   ├── prompts/                    # Markdown prompt files for each agent
│   ├── scholar_agent/              # Research & Q&A agent (LangGraph)
│   ├── study_plan_agent/           # Study plan generation agent (LangGraph)
│   ├── ui/
│   │   └── streamlit_app.py        # Streamlit chat interface
│   └── utils/
│       ├── llm_factory.py          # LLM client factory
│       └── mcp_client.py           # Shared MCP client helper
├── pyproject.toml
└── .env.example
```
