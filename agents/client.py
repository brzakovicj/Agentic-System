from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessageChunk
from typing import AsyncGenerator
from agents.graph import AgentWorkflow, AgentState
from agents.mcp_servers.config import mcp_config

class MCPClient:
    def __init__(self, config: dict):
        self.client = MultiServerMCPClient(connections = config["mcpServers"])

    async def get_tools(self):
        """Fetch tools from all connected MCP servers."""
        try:
            tools = await self.client.get_tools()

            for t in tools:
                print(t.name, t.description)

            return tools
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from MCP servers: {e}")


class StreamProcessor:
    async def process(self, graph, input, config) -> AsyncGenerator[str, None]:
        """
            Stream the response from the graph while parsing out tool calls.

            Args:
                input: The input for the graph.
                graph: The graph to run.
                config: The config to pass to the graph. Required for memory.

            Yields:
                A processed string from the graph's chunked response. 
        """
        
        async for chunk, metadata in graph.astream(
            input = input,
            stream_mode = "messages",
            config = config
        ):
            yield self._parse_chunk(chunk)


    def _parse_chunk(self, chunk):
        """Parse a message chunk from the graph's response, extracting content and tool calls."""

        if not isinstance(chunk, AIMessageChunk):
            return ""
        
        if chunk.response_metadata:
            if chunk.response_metadata.get("finish_reason") == "tool_calls":
                return "\n\n"
            
        if chunk.tool_call_chunks:
            return self._parse_tool_calls(chunk)
            
        return chunk.content or ""

    def _parse_tool_calls(self, chunk):
        """Parse out tool calls from a message chunk and format them for display."""

        result = ""
        for tool in chunk.tool_call_chunks:
            tool_name = tool.get("name", "")
            args = tool.get("args", "")

            if tool_name:
                result += f"\n\n< TOOL CALL: {tool_name} >\n\n"
            if args:
                result += args
        
        return result


class AgentWorkflowRunner:
    def __init__(self, tools):
        self.workflow = AgentWorkflow(tools=tools)

    async def create_graph(self):
        return await self.workflow._create_graph()
    
    async def run(self, graph, user_input, config, streamer):
        """Run the agent workflow with the given graph, user input, and config, streaming the output."""

        input_state = AgentState(messages = [HumanMessage(content = user_input)])

        async for output in streamer.process(graph, input_state, config):
            yield output


async def client_main():
    streamer = StreamProcessor()

    mcp_client = MCPClient(mcp_config)
    tools = await mcp_client.get_tools()

    runner = AgentWorkflowRunner(tools)
    graph = await runner.create_graph()

    # pass a config with a thread_id to use memory
    graph_config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    while True:
        user_input = input("\n\nUSER: ")
        if user_input in ["quit", "exit"]:
            break

        print("\n ----  USER  ---- \n\n", user_input)
        print("\n ----  ASSISTANT  ---- \n\n")

        async for chunk in runner.run(
            graph = graph, 
            user_input = user_input, 
            config = graph_config, 
            streamer = streamer
        ):
            print(chunk, end = "", flush = True)



# async def stream_graph_response(
#         input: AgentState, 
#         graph: StateGraph, 
#         config: dict = {}
#     ) -> AsyncGenerator[str, None]:
#     """
#         Stream the response from the graph while parsing out tool calls.

#         Args:
#             input: The input for the graph.
#             graph: The graph to run.
#             config: The config to pass to the graph. Required for memory.

#         Yields:
#             A processed string from the graph's chunked response.
#     """

#     async for message_chunk, metadata in graph.astream(
#         input=input,
#         stream_mode="messages",
#         config=config
#     ):        
#         if isinstance(message_chunk, AIMessageChunk):
#             if message_chunk.response_metadata:
#                 finish_reason = message_chunk.response_metadata.get("finish_reason", "")
#                 if finish_reason == "tool_calls":
#                     yield "\n\n"

#             if message_chunk.tool_call_chunks:
#                 for tool_chunk in message_chunk.tool_call_chunks:
#                     tool_name = tool_chunk.get("name", "")
#                     args = tool_chunk.get("args", "")
#                     tool_call_str = ""

#                     if tool_name:
#                         tool_call_str += f"\n\n< TOOL CALL: {tool_name} >\n\n"
#                     if args:
#                         tool_call_str += args

#                     yield tool_call_str
#             else:
#                 yield message_chunk.content
#             continue


# async def main():
#     """
#     Initialize the MCP client and run the agent conversation loop.

#     The MultiServerMCPClient allows connection to multiple MCP servers using a single client and config.
#     """
#     print(mcp_config)
#     client = MultiServerMCPClient(connections=mcp_config["mcpServers"])

#     # the get_tools() method returns a list of tools from all the connected servers
#     tools = await client.get_tools()

#     for t in tools:
#        print(t.name, t.description)

#     graph = await AgentWorkflow(tools=tools)._create_graph()

#     # pass a config with a thread_id to use memory
#     graph_config = {
#         "configurable": {
#             "thread_id": "1"
#         }
#     }

#     while True:
#         user_input = input("\n\nUSER: ")
#         if user_input in ["quit", "exit"]:
#             break

#         print("\n ----  USER  ---- \n\n", user_input)
#         print("\n ----  ASSISTANT  ---- \n\n")

#         async for response in stream_graph_response(
#             input = AgentState(messages = [HumanMessage(content=user_input)]),
#             graph = graph, 
#             config = graph_config
#         ):
#             print(response, end="", flush=True)

# if __name__ == "__main__":
#     asyncio.run(main())