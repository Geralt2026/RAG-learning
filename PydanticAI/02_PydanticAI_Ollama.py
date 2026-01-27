from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIChatModel(
    model_name='qwen3:4b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
# 创建Agent对象  Agent对象是PydanticAI的入口，用于执行任务  model是模型，output_type是输出类型
agent = Agent(ollama_model, output_type=CityLocation)

# 运行Agent 执行任务
result = agent.run_sync('Where were the olympics held in 2012?')

# 输出结果 输出的是CityLocation对象  city='London' country='United Kingdom' 是一个Pydantic模型
print(result.output)

# 输出使用情况  使用情况是一个RunUsage对象  input_tokens=411, output_tokens=4863, requests=2
print(result.usage())