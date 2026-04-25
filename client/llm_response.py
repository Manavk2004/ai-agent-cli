from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class TextDelta:
    content: str

    def __str__(self):
        return self.content

class StreamEventType(str, Enum):
    TEXT_DELTA = 'text_delta'
    MESSAGE_COMPLETE = 'message_complete'
    ERROR = 'error'
    TOOL_USE = "tool_use"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )
# a + b

@dataclass
class StreamEvent:
    type: StreamEventType
    tool_use: dict[str, Any] | None = None
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    