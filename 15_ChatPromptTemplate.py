# PromptTemplate：通用提示词模板，支持动态注入信息。
# FewShotPromptTemplate：支持基于模板注入任意数量的示例信息。
# ChatPromptTemplate：支持注入任意数量的历史会话信息。

# 通过from_messages方法，从列表中获取多轮次会话作为聊天的基础模板
# 前面PromptTemplate类用的from_template仅能接入一条消息，而from_messages可以接入一个list的消息

# 历史会话信息并不是静态的（固定的），而是随着对话的进行不停地积攒，即动态的。
# 所以，历史会话信息需要支持动态注入。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "........"),
        ("ai", "........"),
        MessagesPlaceholder("history"), # MessagePlaceholder作为占位, 提供history作为占位的key
        ("human", "........"),
    ]
)

history_data = [
    ("human", "..."), 
    ("ai", "..."), 
    ("human", "..."),
     ("ai", "...")
]

# 基于invoke动态注入历史会话记录
# 必须是invoke，format无法注入
prompt_value = chat_template.invoke({"history": history_data})
