# 让向量检索加入链？
# 使用RunnablePassthrough类

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

prompt = ChatPromptTemplate.from_template(
    """你是一个专业的助手。请利用下方检索到的背景资料来回答问题。
    如果资料中没有答案，请直说不知道，不要胡编乱造。
    资料：{context}
    问题：{input}
    """
)

model = ChatTongyi(model="qwen3-max")

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings())

# 准备一下资料（向量库的数据）
# add_texts 传入一个list[str]
vector_store.add_texts(
    [
        "Python 基础语法包括变量、循环和函数。",
        "RAG 技术可以给大模型提供外部搜索能力。",
        "如何制作一杯好喝的拿铁：需要浓缩咖啡和蒸汽牛奶。",
        "Conda 是一个开源的软件包管理系统。",
    ]
)

input_text = "Python 基础语法包括变量、循环和函数。"

# 检索向量库
result = vector_store.similarity_search(input_text, k=2)
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

chain = prompt | model | StrOutputParser()

res = chain.invoke({"input": input_text, "context": reference_text})
print(res)
