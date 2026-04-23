from anthropic import AsyncAnthropic

_client: AsyncAnthropic | None = None

def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


async def count_tokens(text: str, model: str = 'claude-haiku-4-5-20251001') -> int:
    client = _get_client()
    response = await client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    if response:
        return response.input_tokens
    else:
        return estimate_tokens(message=text)

def estimate_tokens(message: str) -> int:
    return max(1, len(message) // 4)
