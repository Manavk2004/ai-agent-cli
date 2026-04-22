from typing import AsyncGenerator

from agent.events import AgentEvent
from client.llm_client import LLMClient
from client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        # Add user message to context

        async for event in self._agentic_loop():
            yield event

        yield AgentEvent.agent_end()



    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        messages = [{"role": "user", "content": "Hey what is going on"}]
        async for event in self.client.chat_completion(messages, True):
            if event.type == StreamEventType.TEXT_DELTA:
                content = event.text_delta.content
                yield AgentEvent.text_delta(content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error occurred")
                