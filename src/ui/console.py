from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import RunnableConfig
from rich.console import Console
from rich.panel import Panel

from src.multi_agent.supervisor.state import SupervisorState
from src.multi_agent.supervisor.graph import SupervisorAgent

def get_responsive_width(console: Console) -> int:
    """Get responsive width with margins for panels."""
    return min(120, console.size.width - 4) if console.size.width > 10 else 80

# async def stream_graph_responses(
#         input: SupervisorState,
#         graph: StateGraph,
#         console: Console,
#         **kwargs
#     ):
#     """Asynchronously stream the result of the graph run with subgraph support.

#     Args:
#         input: The input to the graph.
#         graph: The compiled graph.
#         console: Rich console for output.
#         **kwargs: Additional keyword arguments.

#     Returns:
#         str: The final LLM or tool call response
#     """
#     # Agent styling configuration
#     AGENT_STYLES = {
#         'researcher': {'color': 'cyan', 'emoji': '🔬', 'name': 'Researcher'},
#         'copywriter': {'color': 'magenta', 'emoji': '✍️', 'name': 'Copywriter'},
#         'supervisor': {'color': 'green', 'emoji': '🎯', 'name': 'Supervisor'},
#     }

#     # Track current AI message source to detect transitions
#     current_ai_source = None
#     current_content = ""
#     current_tool_args = ""
#     current_tool_name = ""

#     async for chunk in graph.astream(
#         input = input,
#         stream_mode = "updates",
#         subgraphs = True,
#         **kwargs
#     ):
#         # Sa subgraphs=True i "updates" struktura je: (namespace, {node_name: state_update})
#         namespace, updates = chunk

#         if isinstance(message_chunk, AIMessageChunk):
#             # Determine the source of this AI message directly from namespace
#             if namespace:
#                 # This is from a subgraph - detect agent from namespace
#                 namespace_str = str(namespace)
#                 if "call_researcher" in namespace_str:
#                     ai_source = "researcher"
#                 elif "call_copywriter" in namespace_str:
#                     ai_source = "copywriter"
#                 else:
#                     # Fallback for unknown subgraphs
#                     ai_source = "researcher"
#             else:
#                 # This is from the main graph (supervisor)
#                 ai_source = "supervisor"

#             # Check if we're transitioning between different AI sources
#             if current_ai_source != ai_source:
#                 # Finalize previous agent's content in a panel
#                 if current_content.strip() and current_ai_source:
#                     style = AGENT_STYLES[current_ai_source]
#                     panel = Panel(
#                         current_content.strip(),
#                         title=f"{style['emoji']} {style['name']}",
#                         border_style=style['color'],
#                         title_align="left",
#                         padding=(1, 2),
#                         width=get_responsive_width(console)
#                     )
#                     console.print(panel)
#                     console.print()  # Add spacing after completed panel

#                 # Start new agent
#                 current_ai_source = ai_source
#                 current_content = ""
#             elif current_ai_source is None:
#                 # First AI message
#                 current_ai_source = ai_source
#                 current_content = ""

#             # Handle tool calls
#             if message_chunk.response_metadata:
#                 finish_reason = message_chunk.response_metadata.get("finish_reason", "")
#                 if finish_reason == "tool_calls":
#                     # Print accumulated tool args if we have them
#                     if current_tool_args.strip():
#                         if current_ai_source:
#                             style = AGENT_STYLES[current_ai_source]
#                             console.print(f"  [dim {style['color']}]{current_tool_args.strip()}[/dim {style['color']}]")
#                         else:
#                             console.print(f"  [dim]{current_tool_args.strip()}[/dim]")
#                         current_tool_args = ""
#                     console.print("  🔧 [yellow]Tool call completed[/yellow]")
#                     console.print()  # Add spacing after tool completion

#             if message_chunk.tool_call_chunks:
#                 tool_chunk = message_chunk.tool_call_chunks[0]
#                 tool_name = tool_chunk.get("name", "")
#                 args = tool_chunk.get("args", "")

#                 if tool_name and tool_name != current_tool_name:
#                     # New tool call - print the name
#                     console.print(f"  🔧 [yellow]TOOL CALL: {tool_name}[/yellow]")
#                     current_tool_name = tool_name
#                     current_tool_args = ""  # Reset args for new tool

#                 if args:
#                     # Accumulate args instead of printing immediately
#                     current_tool_args += args
#             else:
#                 # Just accumulate content for panel display
#                 if message_chunk.content:
#                     current_content += message_chunk.content
#         else:
#             # Handle other message types
#             pass

#     # Print any remaining tool args
#     if current_tool_args.strip():
#         if current_ai_source:
#             style = AGENT_STYLES[current_ai_source]
#             console.print(f"  [dim {style['color']}]{current_tool_args.strip()}[/dim {style['color']}]")
#         else:
#             console.print(f"  [dim]{current_tool_args.strip()}[/dim]")
#         console.print()

#     # Finalize the last agent's content in a panel
#     if current_content.strip() and current_ai_source:
#         style = AGENT_STYLES[current_ai_source]
#         panel = Panel(
#             current_content.strip(),
#             title=f"{style['emoji']} {style['name']}",
#             border_style=style['color'],
#             title_align="left",
#             padding=(1, 2),
#             width=get_responsive_width(console)
#         )
#         console.print(panel)
#         console.print()  # Add spacing after final panel

async def stream_graph_responses(
    input: SupervisorState,
    graph: StateGraph,
    console: Console,
    **kwargs
):
    AGENT_STYLES = {
        'researcher': {'color': 'cyan', 'emoji': '🔬', 'name': 'Researcher'},
        'copywriter': {'color': 'magenta', 'emoji': '✍️', 'name': 'Copywriter'},
        'supervisor': {'color': 'green', 'emoji': '🎯', 'name': 'Supervisor'},
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

async def main():
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

        supervisor = SupervisorAgent()
        graph = await supervisor.build_graph()

        while True:
            console.print()
            user_input = console.input("[bold blue]User:[/bold blue] ")
            console.print()  # Add spacing after user input

            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[yellow]Exit command received. Goodbye! 👋[/yellow]\n")
                break

            graph_input = SupervisorState(
                messages = [ HumanMessage(content=user_input) ]
            )

            await stream_graph_responses(graph_input, graph, console, config = config)

    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}: {str(e)}[/red]")
        raise