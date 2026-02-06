from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.dashscope import DashScope
from agno.learn import (
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)

agent = Agent(
    model=DashScope(
        id="qwen3-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    db=SqliteDb(db_file="tmp/agents.db"),
    markdown=True,
    add_history_to_context=True,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
    ),
)

if __name__ == "__main__":
    user_id = "alice@example.com"

    # Session 1: Agent decides what to save via tool calls
    print("\n--- Session 1: Agent uses tools to save profile and memories ---\n")
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
