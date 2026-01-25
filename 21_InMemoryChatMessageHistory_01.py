# 如果想要封装历史记录，除了自行维护历史消息外，也可以借助LangChain内置的历史记录附加功能
# LangChain提供了History功能，帮助模型在有历史记忆的情况下回答
# 基于RunnableWithMessageHistory在原有链的基础上创建带有历史记录功能的新链（新Runnable实例）
# 基于InMemoryChatMessageHistory为历史记录提供内存存储（临时用）

from langchain_core.runnables.history import RunnableWithMessageHistory

# 通过RunnableWithMessageHistory获取一个新的带有历史记录功能的chain
conversation_chain = RunnableWithMessageHistory(
    some_chain,  # 被附加历史消息的Runnable，通常是chain
    None,  # 获取指定会话ID的历史会话的函数
    input_messages_key="input",  # 声明用户输入消息在模板中的占位符
    history_messages_key="chat_history",  # 声明历史消息在模板中的占位符
)

# 获取指定会话ID的历史会话记录函数
chat_history_store = {}     

# 存放多个会话ID所对应的历史会话记录
# 函数传入为会话ID（字符串类型）
# 函数要求返回BaseChatMessageHistory的子类
# BaseChatMessageHistory类专用于存放某个会话的历史记录
# InMemoryChatMessageHistory是官方自带的基于内存存放历史记录的类

def get_history(session_id):
    if session_id not in chat_history_store:
    # 返回一个新的实例
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

# 完整代码见02