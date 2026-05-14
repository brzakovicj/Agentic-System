import asyncio
import sys
from src.mcp_servers.chromadb_server.server import mcp
from src.ui.console import main as multi_agent_main
from src.utils.llm_factory import LLMFactory
#import nest_asyncio

def main():
    if len(sys.argv) < 2:
        print("Defaulting to client mode. To specify mode, run with ['client', 'server' or 'supervisor'] argument.")
        #nest_asyncio.apply()
        asyncio.run(multi_agent_main())
        return
    
    LLMFactory.initialize()
    
    mode = sys.argv[1]

    if mode == "server":
        mcp.run(transport="stdio")

    elif mode == "supervisor":
        #nest_asyncio.apply()
        asyncio.run(multi_agent_main(mode))

    elif mode == "orchestrator":
        asyncio.run(multi_agent_main(mode))

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
