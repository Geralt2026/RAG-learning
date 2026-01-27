"""
使用 PydanticAI 和阿里云百炼 qwen3-max 模型的示例

配置说明：
1. 设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）
2. 使用 OpenAI 兼容模式连接阿里云百炼

运行方式：
    python 01_HelloPydanticAI.py
"""

import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 定义 Pydantic 模型（输出结构）
class MyModel(BaseModel):
    city: str
    country: str

# 配置阿里云百炼的 OpenAI 兼容 API
# API Key 从环境变量读取（DASHSCOPE_API_KEY 或 OPENAI_API_KEY）
api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        "请设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）\n"
        "Windows: set DASHSCOPE_API_KEY=your-api-key\n"
        "Linux/Mac: export DASHSCOPE_API_KEY=your-api-key"
    )

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

print(f"✅ 使用模型: qwen3-max (阿里云百炼)")

# 创建 Agent 对象
# Agent 是 PydanticAI 的入口，用于执行任务
# model 是模型，output_type 指定输出类型为 MyModel MyModel是一个Pydantic模型
agent = Agent(model, output_type=MyModel)

# 运行 Agent
if __name__ == "__main__":
    result = agent.run_sync("The windy city in the US of A.")
    
    # 输出结果
    print(f"\n📊 解析结果:")
    # 输出结果 输出的是MyModel对象 city='Chicago' country='United States' 是一个Pydantic模型
    print(result.output)
    
    # 输出使用情况
    usage = result.usage()
    if usage:
        print(f"\n📈 使用情况:")
        print(f"  {usage}")
