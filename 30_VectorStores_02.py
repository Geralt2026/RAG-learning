# 外部（Chroma）向量存储的使用
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",  # 集合名称
    embedding_function=DashScopeEmbeddings(),  # 嵌入函数
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


# 向量存储类均提供3个通用API接口：
# add_document，添加文档到向量存储
# delete，从向量存储中删除文档
# similarity_search：相似度搜索