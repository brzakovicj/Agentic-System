from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPClient:
    def __init__(self, config: dict):
        # self.client = MultiServerMCPClient(connections = config["mcpServers"])
        self.config = config["mcpServers"]
        self.client = MultiServerMCPClient(connections=self.config)
        self._tools = None
        self._sessions = []

    async def get_tools(self):
        if self._tools is not None:
            return self._tools

        all_tools = []
        for server_name in self.config:
            session_cm = self.client.session(server_name)
            session = await session_cm.__aenter__()
            self._sessions.append((session_cm, session))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

        self._tools = all_tools
        return self._tools

    async def close(self):
        for session_cm, session in self._sessions:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions = []
        self._tools = None