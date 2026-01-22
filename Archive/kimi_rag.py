import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 配置 Kimi API (Kimi 兼容 OpenAI 格式)
client = ChatOpenAI(
    api_key="sk-5RGxWeXho8KGYYMtQEsfGYX0mBbRZTHRmoLOffV608DjbnRk", 
    base_url="https://api.moonshot.cn/v1",
    model_name="moonshot-v1-8k"
)

# 2. 模拟 RAG 检索回来的“背景知识”
# 实际开发中，这里是通过向量数据库检索出来的
context = "根据最新规定，Python 学习者在掌握基础语法后，应优先通过 RAG 技术构建个人知识库。"
question = "刚学完 Python 应该做什么？"

# 3. 编写 RAG 专用 Prompt
template = """你是一个专业的 AI 导师。请根据提供的上下文(Context)来回答问题(Question)。
如果上下文中没有提到相关信息，请诚实回答不知道。

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. 组合并运行
chain = prompt | client
response = chain.invoke({"context": context, "question": question})

print("Kimi 的回答：")
print(response.content)