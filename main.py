import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.groq import GroqService

from core.cli_chat import CliChat
from core.cli import CliApp

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# Groq configuration
groq_model = os.getenv("GROQ_MODEL", "")
groq_api_key = os.getenv("GROQ_API_KEY", "")


if not groq_model:
    raise ValueError("GROQ_MODEL cannot be empty. Set it in mcp/.env.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY cannot be empty. Set it in mcp/.env.")


async def main():
    groq_service = GroqService(model=groq_model, api_key=groq_api_key)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            groq_service=groq_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
