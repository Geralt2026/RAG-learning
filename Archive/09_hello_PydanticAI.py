from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# 初始化本地 Ollama 模型
ollama_provider = OllamaProvider(base_url="http://localhost:11434/v1")

ollama_model = OpenAIChatModel(
    model_name="deepseek-r1:8b",
    provider=ollama_provider,
)

agent = Agent(  
    ollama_model,
    instructions='Be concisely, reply with one sentence.',  
)

result = agent.run_sync('Where were the olympics held in 2012?')  
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""