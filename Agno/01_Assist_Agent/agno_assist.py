import asyncio
import os
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.dashscope import DashScope
from agno.vectordb.lancedb import LanceDb, SearchType

# Create embedder
embedder = OpenAIEmbedder(
    id="text-embedding-v4",  # DashScope embedding model
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Create knowledge base with hybrid search
knowledge = Knowledge(
    vector_db=LanceDb(
        path="Agno/01_Assist_Agent/data/lancedb",
        table_name="agno_assist_knowledge",
        search_type=SearchType.hybrid,  # Semantic + keyword search
        embedder=embedder,
    ),
)

# Load documentation asynchronously
asyncio.run(
    knowledge.ainsert(name="Agno Docs", url="https://docs.agno.com/llms-full.txt")
)

# Create agent with knowledge and session persistence
agno_assist = Agent(
    name="Agno Assist",
    model=DashScope(
        id="qwen3-max", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    description="You help answer questions about the Agno framework.",
    instructions="Search your knowledge before answering the question.",  # Forces knowledge search
    knowledge=knowledge,
    db=SqliteDb(  # Stores conversation history
        session_table="agno_assist_sessions", db_file="tmp/agents.db"
    ),
    add_history_to_context=True,
    add_datetime_to_context=True,
    markdown=True,
)

if __name__ == "__main__":
    agno_assist.print_response("What is Agno?")
    agno_assist.print_response("How do I create an agent with tools?")
    agno_assist.print_response("What vector databases does Agno support?")
