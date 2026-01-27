"""
配置模块 - 向后兼容层
推荐使用 config_loader.py 和 config.yaml（更符合企业规范）

使用方式：
1. 新项目：from config_loader import config
2. 旧代码：import config_data as config（继续工作）
"""
try:
    # 优先使用 YAML 配置（推荐）
    from config_loader import config as yaml_config
    if yaml_config is not None:
        # YAML 配置可用，使用 YAML 配置
        config = yaml_config
    else:
        raise ImportError("YAML 配置不可用")
except (ImportError, FileNotFoundError):
    # 如果 YAML 配置不可用，回退到旧配置
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
    
    # 元数据配置（旧方式）
    metadata_operator = "Beamus Wayne"
