import asyncio
from typing import Any, AsyncGenerator

from anthropic import APIConnectionError, APIError, AsyncAnthropic, RateLimitError
from dotenv import load_dotenv
import os

from client.response import EventType, StreamEvent, TextDelta, TokenUsage

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None
        self._max_retries: int = 3

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
    ) -> AsyncGenerator[StreamEvent, None]:
        
        client = self.get_client()

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5-20251001",
            "messages": non_system,
            "max_tokens": max_tokens,
        }
    
        for attempt in range(self._max_retries + 1):
            try:
                if system_parts:
                    kwargs["system"] = "\n\n".join(system_parts)


                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event


                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error= f"rate limti exceeded: {e}"
                    )
                    return

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error= f"Connection error: {e}"
                    )
                    return

            except APIError as e:
                yield StreamEvent(
                    type=EventType.ERROR,
                    error= f"Connection error: {e}"
                )

                return







    async def _stream_response(
        self,
        client: AsyncAnthropic,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.messages.create(**kwargs, stream=True)

        input_tokens = 0
        cached_tokens = 0
        output_tokens = 0
        stop_reason = None

        async for chunk in response:
            if chunk.type == "message_start":
                input_tokens = chunk.message.usage.input_tokens
                cached_tokens = chunk.message.usage.cache_read_input_tokens or 0
            elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield StreamEvent(
                    type=EventType.TEXT_DELTA,
                    text_delta=TextDelta(content=chunk.delta.text),
                )
            elif chunk.type == "message_delta":
                output_tokens = chunk.usage.output_tokens
                stop_reason = chunk.delta.stop_reason

        yield StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            finish_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=cached_tokens,
            ),
        )




    async def _non_stream_response(
        self,
        client: AsyncAnthropic,
        kwargs: dict[str, Any],
    ) -> StreamEvent:
        response = await client.messages.create(**kwargs)
        message = response.content[0].text

        text_delta = None
        if message:
            text_delta = TextDelta(content=message)
        
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cached_tokens=response.usage.cache_read_input_tokens
            )

        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=response.stop_reason,
            usage=usage,
        )
    
    

