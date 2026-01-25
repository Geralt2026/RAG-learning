from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# 配置信息
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template("你需要根据对话历史回应用户问题。对话历史：{chat_history}。用户当前输入：{input}， 请给出回应")
str_parser = StrOutputParser()
store = {}


# 通过会话id获取InMemoryChatMessageHistory对象
def get_history(session_id):
    if session_id not in store:
        # 存入新的实例
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 基链
base_chain = prompt | model | StrOutputParser()


# 增强链：自动附加历史信息
conversation_chain = RunnableWithMessageHistory(
    base_chain,    # 被增强的原有链
    get_history,   # 通过会话id获取InMemoryChatMessageHistory对象
    input_messages_key="input",         # 声明用户输入消息在模板中的占位符
    history_messages_key="chat_history" # 声明历史消息在模板中的占位符
)

if __name__ == '__main__':
    # 固定格式，添加LangChain的配置，为当前程序配置所属的Session_id
    session_config = {
        "configurable": {
            "session_id": "user_001"
            }
        }

    print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))