from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os  
import logfire

# 配置日志
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

# 创建Agent
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))

# 创建Agent
agent = Agent(model, instructions='Be fun!')
app = agent.to_ag_ui()