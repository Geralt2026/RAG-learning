# 使用InMemoryChatMessageHistory仅可以在内存中临时存储会话记忆，一旦程序退出，则记忆丢失。
# InMemoryChatMessageHistory 类继承自 BaseChatMessageHistory

# FileChatMessageHistory类实现，核心思路：
# 基于文件存储会话记录，以session_id为文件名，不同session_id有不同文件存储消息
# 继承BaseChatMessageHistory实现如下3个方法：
# add_messages:同步模式，添加消息
# messages:同步模式，获取消息
# clear：同步模式，清除消息

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseMessage, InMemoryChatMessageHistory,BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import messages_from_dict, message_to_dict
# message_to_dict 单个消息对象（BaseMessge类） → 字典
# messages_from_dict：【字典，字典...】 → 【消息，消息...】
# AIMessage、HumanMessage、SystemMessage都是BaseMessage的子类

from typing import Sequence, List
import json
import os

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id  # 会话ID
        self.storage_path = storage_path  # 不同会话id的存储文件，所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        #确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence[BaseMessage] 消息序列，类似于list、tuple
        all_messages = list(self.messages) #已有的消息列表
        all_messages.extend(messages) # 添加新消息

        # 将数据同步写入到本地文件中
        # 类对象写入文件 → 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        # 官方提供了message_to_dict 单个消息对象（BaseMessge类） → 字典

        # new_messages = []
        # for message in all_messages:
            # message_dict = message_to_dict(message)
            # new_messages.append(message_dict)
        # 列表推导式:
        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)
    
    @property # 属性装饰器，将方法变成成员属性
    def messages(self) -> list[BaseMessage]: 
        # 当前文件内：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data) # 返回值是list[BaseMessage]
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


# 配置信息
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template("你需要根据对话历史回应用户问题。对话历史：{chat_history}。用户当前输入：{input}， 请给出回应")
str_parser = StrOutputParser()



# 通过会话id获取InMemoryChatMessageHistory对象
def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


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

    # print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    # print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))