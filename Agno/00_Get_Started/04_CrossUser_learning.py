import os
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.dashscope import DashScope
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.vectordb.chroma import ChromaDb, SearchType

embedder = OpenAIEmbedder(
    id="text-embedding-v4",  # DashScope embedding model
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

knowledge = Knowledge(
    name="Agent Learnings",
    vector_db=ChromaDb(
        name="learnings",
        path="tmp/chromadb",
        persistent_client=True,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

agent = Agent(
    model=DashScope(
        id="qwen3-max", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    db=SqliteDb(db_file="tmp/agents.db"),
    add_history_to_context=True,
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    markdown=True,
)

if __name__ == "__main__":
    # Session 1: User 1 teaches the agent
    print("\n--- Session 1: User 1 saves a learning ---\n")
    agent.print_response(
        "We're trying to reduce our cloud egress costs. Remember this.",
        user_id="engineer_1@example.com",
        session_id="session_1",
        stream=True,
    )

    lm = agent.get_learning_machine()
    lm.learned_knowledge_store.print(query="cloud")

    # Session 2: User 2 benefits from the learning
    print("\n--- Session 2: User 2 asks a related question ---\n")
    agent.print_response(
        "I'm picking a cloud provider for a data pipeline. Key considerations?",
        user_id="engineer_2@example.com",
        session_id="session_2",
        stream=True,
    )
