from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import OllamaResponses


ollama_model = OllamaResponses(id="qwen3-vl:4b")

agent = Agent(
    model=ollama_model,
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
)

agent.print_response("Hi, I am Katya. I work for the CIA. What is the capital of France?", stream=True)