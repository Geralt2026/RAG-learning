from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.dashscope import DashScope

agent = Agent(
    model=DashScope(
        id="qwen3-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
    markdown=True,
    add_history_to_context=True,
)

if __name__ == "__main__":
    user_id = "alice@example.com"

    # Session 1: Share information naturally
    print("\n--- Session 1: Extraction happens automatically ---\n")
    agent.print_response(
        "Hi! I'm Alice. I work at Anthropic as a research scientist. "
        "I prefer concise responses without too much explanation.",
        user_id=user_id,
        session_id="session_1",
        stream=True,
    )

    lm = agent.get_learning_machine()
    lm.user_profile_store.print(user_id=user_id)
    lm.user_memory_store.print(user_id=user_id)

    # Session 2: New session - agent remembers
    print("\n--- Session 2: Agent remembers across sessions ---\n")
    agent.print_response(
        "What do you know about me?",
        user_id=user_id,
        session_id="session_2",
        stream=True,
    )