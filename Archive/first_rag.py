from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DeterministicFakeEmbedding

# 1. 模拟一些私有知识（你的“书”）
texts = [
    "Python 基础语法包括变量、循环和函数。",
    "RAG 技术可以给大模型提供外部搜索能力。",
    "如何制作一杯好喝的拿铁：需要浓缩咖啡和蒸汽牛奶。",
    "Conda 是一个开源的软件包管理系统。"
]

# 2. 初始化一个“假”的嵌入模型（仅用于本地学习原理，不需要网络）
embeddings = DeterministicFakeEmbedding(size=1536)

# 3. 创建向量数据库并存入数据
# 这步就像是给书做了索引：文字 -> 向量 -> 存入数据库
vectorstore = Chroma.from_texts(texts, embeddings)

# 4. 模拟用户提问
query = "我想喝咖啡，该怎么做？"

# 5. 执行检索：寻找和问题语义最接近的片段
docs = vectorstore.similarity_search(query, k=1)

print(f"用户问题：{query}")
print(f"检索到的最相关资料：{docs[0].page_content}")