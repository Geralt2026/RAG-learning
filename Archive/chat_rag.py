from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 初始化模型
llm = ChatOllama(model="qwen3:0.6b", temperature=0.7) 

# 2. 定义 Prompt 模板（关键：加入 MessagesPlaceholder）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 导师。请根据对话历史回答问题。"),
    MessagesPlaceholder(variable_name="history"), # 这里就是存放记忆的地方
    ("human", "{question}")
])

# 3. 构建链
chain = prompt | llm | StrOutputParser()

# 4. 模拟一个简单的内存记忆字典
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 5. 包装成带记忆的链
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --- 模拟对话测试 ---
session_config = {"configurable": {"session_id": "user_001"}}

print("--- 第一轮 ---")
ans1 = with_message_history.invoke(
    {"question": "你好，我是 Python 初学者，我叫小明。"}, 
    config=session_config
)
print(f"AI: {ans1}")

print("\n--- 第二轮 ---")
ans2 = with_message_history.invoke(
    {"question": "你还记得我叫什么名字吗？"}, 
    config=session_config
)
print(f"AI: {ans2}")