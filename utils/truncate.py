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
    return estimate_tokens(text)


def estimate_tokens(text: str, model: str | None = None) -> int:
    return max(1, len(text) // 4)


def truncate_text(
    text: str,
    model: str,
    max_tokens: int,
    suffix: str = "\n... [truncated]",
    preserve_lines: bool = True,
):
    current_tokens = estimate_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    suffix_tokens = estimate_tokens(suffix, model)
    target_tokens = max_tokens - suffix_tokens

    if target_tokens <= 0:
        return suffix.strip()

    if preserve_lines:
        return _truncate_by_lines(text, target_tokens, suffix, model)
    else:
        return _truncate_by_chars(text, target_tokens, suffix, model)


def _truncate_by_lines(text: str, target_tokens: int, suffix: str, model: str) -> str:
    lines = text.split("\n")
    result_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line + "\n", model)
        if current_tokens + line_tokens > target_tokens:
            break
        result_lines.append(line)
        current_tokens += line_tokens

    if not result_lines:
        return _truncate_by_chars(text, target_tokens, suffix, model)

    return "\n".join(result_lines) + suffix


def _truncate_by_chars(text: str, target_tokens: int, suffix: str, model: str) -> str:
    low, high = 0, len(text)

    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid], model) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    return text[:low] + suffix
