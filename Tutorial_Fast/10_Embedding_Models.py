# Embeddings Models
from langchain_community.embeddings import DashScopeEmbeddings

#初始化嵌入模型对象，其默认使用模型是：text-embedding-v1
embed = DashScopeEmbeddings()

#测试
print(embed.embed_query("我喜欢你")) # 单次转换
print(embed.embed_documents(["我喜欢你", "我爱你", "我想你"])) # 批量转换