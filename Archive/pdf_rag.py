import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 配置 Kimi 模型 (使用 OpenAI 兼容模式)
llm = ChatOpenAI(
    api_key="sk-5RGxWeXho8KGYYMtQEsfGYX0mBbRZTHRmoLOffV608DjbnRk",
    base_url="https://api.moonshot.cn/v1",
    model_name="moonshot-v1-8k"
)

# 2. 加载 PDF 文档并切分
loader = PyPDFLoader("test.pdf")
docs = loader.load()

# 将长文档切分成 500 字的小块，重叠 50 字以保持语境连贯
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. 创建本地向量库 (暂使用 FakeEmbedding 演示，若有 OpenAI Key 可换成真正的 Embedding)
# 注意：这里为了方便你直接运行，使用了本地缓存模拟。
# 如果你有 OpenAI 的 Embedding Key，建议将下面换成 OpenAIEmbeddings()
from langchain_community.embeddings import DeterministicFakeEmbedding
vectorstore = Chroma.from_documents(documents=splits, embedding=DeterministicFakeEmbedding(size=1536))

# 4. 构建检索器 (Retriever)
retriever = vectorstore.as_retriever()

# 5. 定义 RAG 流程控制
template = """你是一个专业的助手。请利用下方检索到的背景资料来回答问题。
如果资料中没有答案，请直说不知道，不要胡编乱造。
资料：{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 核心链式结构：检索 -> 格式化 -> 填入 Prompt -> 调用模型 -> 输出结果
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 测试问答
question = "这份文档主要讲了什么内容？"
print(f"Question: {question}")
print(f"Answer: {rag_chain.invoke(question)}")