import asyncio

from agents.client import main as client_main

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py [client|server]")
    #     return

    # mode = sys.argv[1]

    # if mode == "client":
    asyncio.run(client_main())

    # elif mode == "server":
    #     from agents.mcp_servers.rag_server.RAG import mcp
    #     mcp.run(transport="stdio")

    # else:
    #     print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
