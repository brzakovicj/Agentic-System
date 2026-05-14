from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import RunnableConfig
from rich.console import Console
from rich.panel import Panel

from src.scheduler_agent.orchestrator.graph import OrchestratorAgent
from src.scheduler_agent.orchestrator.state import OrchestratorState
from src.researcher_agent.supervisor.state import SupervisorState
from src.researcher_agent.supervisor.graph import SupervisorAgent

def get_responsive_width(console: Console) -> int:
    """Get responsive width with margins for panels."""
    return min(120, console.size.width - 4) if console.size.width > 10 else 80

async def stream_graph_responses(
    input: SupervisorState,
    graph: StateGraph,
    console: Console,
    **kwargs
):
    AGENT_STYLES = {
        'researcher': {'color': 'cyan', 'emoji': '🔬', 'name': 'Researcher'},
        'copywriter': {'color': 'magenta', 'emoji': '✍️', 'name': 'Copywriter'},
        'scheduler': {'color': 'cyan', 'emoji': '🔬', 'name': 'Scheduler'},
        'supervisor': {'color': 'green', 'emoji': '🎯', 'name': 'Supervisor'},
        'orchestrator': {'color': 'green', 'emoji': '🎯', 'name': 'Orchestrator'},
    }

    async for chunk in graph.astream(
        input=input,
        stream_mode="updates",
        subgraphs=True,
        **kwargs
    ):
        # Sa subgraphs=True i "updates" struktura je: (namespace, {node_name: state_update})
        namespace, updates = chunk

        # Odredi koji agent je u pitanju
        if namespace:
            ns_str = str(namespace)
            if "call_researcher" in ns_str:
                agent_key = "researcher"
            elif "call_copywriter" in ns_str:
                agent_key = "copywriter"
            elif "call_scheduler" in ns_str:
                agent_key = "scheduler"
            else:
                agent_key = "researcher"
        else:
            agent_key = "supervisor"

        style = AGENT_STYLES[agent_key]

        # updates je dict: {node_name: {"messages": [...]}}
        for node_name, state_update in updates.items():
            messages = state_update.get("messages", [])

            for msg in messages:
                content = ""

                if isinstance(msg, AIMessage):
                    # Prikaži tool calls ako postoje
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            console.print(
                                f"  🔧 [yellow]TOOL CALL [{node_name}]: "
                                f"{tc['name']}[/yellow]"
                            )
                            console.print(
                                f"  [dim yellow]{tc['args']}[/dim yellow]"
                            )
                        console.print()
                    
                    if msg.content:
                        content = msg.content

                elif isinstance(msg, ToolMessage):
                    # Opcionalno prikaži tool rezultate
                    console.print(
                        f"  [dim {style['color']}]"
                        f"[Tool result: {msg.name}][/dim {style['color']}]"
                    )
                    console.print()

                if content.strip():
                    panel = Panel(
                        content.strip(),
                        title=f"{style['emoji']} {style['name']} [{node_name}]",
                        border_style=style['color'],
                        title_align="left",
                        padding=(1, 2),
                        width=get_responsive_width(console)
                    )
                    console.print(panel)
                    console.print()

async def main(mode: str):
    """Main function to run the supervisor with subgraphs."""
    # Create console without fixed width - let it be responsive
    console = Console()

    try:
        config = RunnableConfig(configurable={
            "thread_id": "1",
            "recursion_limit": 50,
        })

        # Welcome panel with responsive width
        welcome_panel = Panel(
            "Multi-Agent Supervisor with Subgraphs\nType 'exit' or 'quit' to stop",
            title="🚀 AI Launchpad",
            border_style="blue",
            title_align="center",
            padding=(1, 2),  # Add padding to welcome panel
            width=get_responsive_width(console)
        )
        console.print(welcome_panel)
        console.print()  # Add spacing after welcome

        if (mode == "supervisor"):
            supervisor = SupervisorAgent()
            graph = await supervisor.build_graph()
        elif (mode == "orchestrator"):
            orchestrator = OrchestratorAgent()
            graph = await orchestrator.build_graph()

        while True:
            console.print()
            user_input = console.input("[bold blue]User:[/bold blue] ")
            console.print()  # Add spacing after user input

            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[yellow]Exit command received. Goodbye! 👋[/yellow]\n")
                break
                
            if (mode == "supervisor"):
                graph_input = SupervisorState(
                    messages = [ HumanMessage(content=user_input) ],
                    final_answer = False
                )
            elif (mode == "orchestrator"):
                graph_input = OrchestratorState(
                    messages = [ HumanMessage(content=user_input) ],
                    final_answer = False
                )

            await stream_graph_responses(graph_input, graph, console, config = config)

    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}: {str(e)}[/red]")
        raise