from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPClient:
    def __init__(self, config: dict):
        self.client = MultiServerMCPClient(connections = config["mcpServers"])

    async def get_tools(self):
        """Fetch tools from all connected MCP servers."""
        try:
            tools = await self.client.get_tools()

            return tools
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from MCP servers: {e}")