from typing import Any

from groq import Groq


class GroqService:
    """Small wrapper around Groq chat completions used by the chat loop."""

    def __init__(self, model: str, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 1.0,
    ):
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 8000,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        return self.client.chat.completions.create(**params)

    @staticmethod
    def text_from_message(message: Any) -> str:
        return message.content or ""