from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化本地模型 (确保你已经执行过 ollama pull qwen3:0.6b)
llm = ChatOllama(
    model="qwen3:0.6b", # 或者你下载的其他模型名，如 "qwen3:0.6b"
    temperature=0
)

# 2. 模拟检索到的上下文
context = "根据最新规定，Python 学习者在掌握基础语法后，应优先通过 RAG 技术构建个人知识库。"
question = "刚学完 Python 应该做什么？"

# 3. 编写 Prompt
template = """你是一个专业的 AI 导师。请根据提供的上下文(Context)来回答问题(Question)。
如果上下文中没有提到相关信息，请诚实回答不知道。

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. 运行
chain = prompt | llm | StrOutputParser()

print("Ollama 的回答：")
# invoke 会触发本地推理，风扇可能会响一下 :)
print(chain.invoke({"context": context, "question": question}))