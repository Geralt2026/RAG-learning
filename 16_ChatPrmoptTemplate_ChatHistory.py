from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "给出每个单词的反义词"),
        # 存储多轮对话的历史记录 history是占位符名称,后续从字典按history作为key取value替代内容
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

model = ChatTongyi(model="qwen3-max")

# StrOutputParser() LangChain内置的结果解析器,可以直接提取结果文本内容,剔除其余元数据信息
chain = prompt | model | StrOutputParser()

# 无历史会话的提问
for chunk in chain.stream(input={"history": [], "question": "粗"}):
    print(chunk)

print("*" * 20)

# 带有历史的提问用HumanMessage（用户消息）和AIMessage（模型消息）封装历史对话
history = [
    # history 要求是一个列表,内部封装用户和AI的对话记录
    HumanMessage(content="开心"),  # 对应("human", "开心")
    AIMessage(content="难过"),  # 对应("ai", "难过")
    HumanMessage(content="高"),  # 对应 ("human", "高")
    AIMessage(content="矮"),  # 对应 ("ai", "矮")
]

# 简化写法,元组的第一个元素是角色(标准角色名human ai) 第二个元素是消息
# history = [
#     ("human", "开心"), ("ai", "难过"),
#     ("human", "高"), ("ai", "矮")
# ]

for chunk in chain.stream(input={"history": history, "question": "粗"}):
    print(chunk)
