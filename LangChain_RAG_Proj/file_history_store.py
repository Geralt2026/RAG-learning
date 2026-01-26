import json
import os
from typing import Sequence
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from langchain_core.chat_history import BaseChatMessageHistory

def get_history(session_id):
    return FileChatMessageHistory(session_id, "LangChain_RAG_Proj\chat_history")

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
