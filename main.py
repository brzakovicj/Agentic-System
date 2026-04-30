import asyncio
import sys
from src.client import client_main
from src.mcp_servers.rag_server.RAG import mcp
from src.multi_agent.supervisor import main as supervisor_main
#import nest_asyncio

def main():
    if len(sys.argv) < 2:
        print("Defaulting to client mode. To specify mode, run with 'client' or 'rag_server' argument.")
        #nest_asyncio.apply()
        asyncio.run(supervisor_main())
        return
    
    mode = sys.argv[1]

    if mode == "client":
        asyncio.run(client_main())

    elif mode == "rag_server":
        mcp.run(transport="stdio")

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
