from agno.agent import Agent
from agno.models.ollama import OllamaResponses

agent = Agent(model=OllamaResponses(id="qwen3-vl:4b"))

agent.print_response(
    "Hi! I'm Alice. I work at Anthropic as a research scientist.", stream=True
)
