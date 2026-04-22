from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import os

from client.response import TextDelta, TokenUsage

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None

    def get_client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stream: bool = True,
        max_tokens: int = 1024,
    ):
        client = self.get_client()

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5-20251001",
            "messages": non_system,
            "max_tokens": max_tokens,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        if stream:
            return await self._stream_response(client, kwargs)
        else:
            return await self._non_stream_response(client, kwargs)

    async def _stream_response(
        self,
        client: AsyncAnthropic,
        kwargs: dict[str, Any],
    ):
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            print()
            return await stream.get_final_message()

    async def _non_stream_response(
        self,
        client: AsyncAnthropic,
        kwargs: dict[str, Any],
    ):
        response = await client.messages.create(**kwargs)
        message = response.content[0].text

        text_delta = None
        if message:
            text_delta = TextDelta(content=message)
        
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cached_tokens=response.usage.cache_read_input_tokens
            )

