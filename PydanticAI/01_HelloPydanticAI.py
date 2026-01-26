import os
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 配置阿里云百炼的 OpenAI 兼容 API
# API Key 从环境变量读取（DASHSCOPE_API_KEY 或 OPENAI_API_KEY）
api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')

# 创建自定义的 OpenAI 客户端，使用阿里云百炼的兼容端点
client = AsyncOpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=api_key
)

# 创建模型，使用阿里云百炼的 qwen3-max
model = OpenAIChatModel(
    'qwen3-max',
    provider=OpenAIProvider(openai_client=client)
)

# 创建 Agent
agent = Agent(
    model,
    instructions='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""