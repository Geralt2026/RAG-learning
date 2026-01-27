md5_path = "LangChain_RAG_Proj\data\md5.txt"

# Chroma
chroma_collection_name = "rag"
chroma_persist_directory = "LangChain_RAG_Proj\data\chroma_db"

# 文本切分
chunk_size = 1000
chunk_overlap = 50
separators = ["\n\n", "\n", "。", "，", "？", "！", "：", "；", "、", "|", " "]
max_split_char_number = 1000  # 文本分隔阈值

# 相似度检索阈值
similarity_threshold = 1  # 检索返回匹配文档的数量

# 嵌入模型
embedding_model = "text-embedding-v4"
chat_model = "qwen3-max"

session_config = {"configurable": {"session_id": "user_001"}}  # 会话配置
