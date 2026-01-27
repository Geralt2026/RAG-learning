import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

model = OpenAIChatModel("qwen3-max", provider=OpenAIProvider(openai_client=client))

agent = Agent(model)  


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:  
        async for message in result.stream_text():  
            print(message)

if __name__ == "__main__":
    asyncio.run(main())