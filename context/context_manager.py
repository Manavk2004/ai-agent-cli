from typing import Any

from config.config import Config
from prompts.system_prompt import get_system_prompt
from dataclasses import dataclass

from utils.truncate import estimate_tokens


@dataclass
class MessageItem:
    role: str
    content: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result['content'] = self.content

        return result

class ContextManager:
    def __init__(self) -> None:
        self._system_prompt = get_system_prompt(Config())
        self._model_name="claude-haiku-4-5-20251001"
        self._messages: list[MessageItem] = []

    def add_user_message(self, content: str) -> None:
        item = MessageItem(
            role='user',
            content=content,
            token_count=estimate_tokens(content)
        )

        self._messages.append(item)


    def add_assistant_message(self, content: str | None) -> None:
        item = MessageItem(
            role='assistant',
            content=content or "",
            token_count=estimate_tokens(content or "")
        )

        self._messages.append(item)


    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append({
                'role': 'system',
                'content': self._system_prompt
            })

        for item in self._messages:
            messages.append(item.to_dict())

        return messages
