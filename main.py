from client.llm_client import LLMClient
import asyncio

async def main():
    client = LLMClient()
    messages = [{
        'role': 'user',
        'content': "What's up"
    }]
    await client.chat_completion(messages, False)
    print("Done")

asyncio.run(main())