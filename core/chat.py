from typing import Any

from core.groq import GroqService
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, groq_service: GroqService, clients: dict[str, MCPClient]):
        self.groq_service = groq_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[dict[str, Any]] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.groq_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )
            assistant_message = response.choices[0].message
            self.messages.append(assistant_message.model_dump(exclude_none=True))

            if assistant_message.tool_calls:
                response_text = self.groq_service.text_from_message(assistant_message)
                if response_text:
                    print(response_text)
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, assistant_message.tool_calls
                )
                self.messages.extend(tool_result_parts)
            else:
                final_text_response = self.groq_service.text_from_message(
                    assistant_message
                )
                break

        return final_text_response
