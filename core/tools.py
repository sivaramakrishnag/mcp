import json
from typing import Any, Optional
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        """Gets all tools from the provided clients."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            print("Available MCP tools:", [t.name for t in tool_models])
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_call_id: str,
        text: str,
    ) -> dict[str, str]:
        """Builds a Groq-compatible tool response message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], tool_calls: list[Any]
    ) -> list[dict[str, str]]:
        """Executes a list of tool requests against the provided clients."""
        tool_result_blocks: list[dict[str, str]] = []
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            print(f"MCP Tool Called: {tool_name}")

            try:
                tool_input = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_call_id,
                        json.dumps({"error": f"Invalid tool arguments: {exc}"}),
                    )
                )
                continue

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_call_id, "Could not find that tool"
                    )
                )
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                if tool_output and tool_output.isError:
                    content_json = json.dumps({"error": content_list})
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                content_json = json.dumps({"error": error_message})

            tool_result_blocks.append(
                cls._build_tool_result_part(tool_call_id, content_json)
            )
        return tool_result_blocks
