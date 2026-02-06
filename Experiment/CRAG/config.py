# -*- coding: utf-8 -*-
"""
CRAG 项目配置
- 知识库路径、向量库、检索/精炼/动作触发阈值
- LLM 与 Embedding 模型（支持 Ollama 与 OpenAI 等）
"""
import os

# 路径配置：以本文件所在目录为基准，使用绝对路径避免受工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 知识库 PDF 存放目录，用户可在此目录下手动添加 PDF（Streamlit/API 任意 cwd 下均一致）
KNOWLEDGE_FILES_DIR = os.path.normpath(os.path.join(BASE_DIR, "Files"))
# 向量库持久化目录
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
# 向量库集合名
CHROMA_COLLECTION_NAME = "crag_kb"

# 检索与精炼
RETRIEVE_TOP_K = 5          # 检索返回的文档数
STRIP_TOP_K = 5             # 知识精炼时保留的条数（按分数）
STRIP_FILTER_THRESHOLD = -0.5   # 精炼时 strip 分数低于此则过滤
CHUNK_SIZE = 800            # 文本切分块大小（字符）
CHUNK_OVERLAP = 80          # 切分重叠

# 动作触发阈值（论文：上阈值、下阈值）
# 任一文档 score > 上阈值 -> Correct；全部 < 下阈值 -> Incorrect；否则 Ambiguous
THRESHOLD_UPPER = 0.5
THRESHOLD_LOWER = -0.9

# LLM 统一使用 qwen3-max（通义千问）；Embedding 可单独配置
LLM_MODEL = os.environ.get("CRAG_LLM_MODEL", "qwen3-max")
LLM_PROVIDER = os.environ.get("CRAG_LLM_PROVIDER", "dashscope")
EMBEDDING_PROVIDER = os.environ.get("CRAG_EMBEDDING_PROVIDER", "dashscope")

# Ollama（仅当 LLM_PROVIDER=ollama 时使用；本地无 qwen3-max 时可设为 qwen3:4b 等）
OLLAMA_LLM_MODEL = os.environ.get("CRAG_OLLAMA_LLM", "qwen3:4b")
OLLAMA_EMBED_MODEL = os.environ.get("CRAG_OLLAMA_EMBED", "nomic-embed-text")

# 通义 / OpenAI 兼容 API（dashscope 或 openai 时使用）
OPENAI_API_KEY = os.environ.get("DASHSCOPE_API_KEY", os.environ.get("DASH_API_KEY", ""))
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")  # OpenAI 兼容 base_url，通义可不填
OPENAI_LLM_MODEL = LLM_MODEL
OPENAI_EMBED_MODEL = os.environ.get("CRAG_EMBED_MODEL", "text-embedding-v3")

# MinerU 模型（用于 PDF 版面分析与文本识别，强制使用）
MINERU_MODEL_ID = "OpenDataLab/MinerU2.5-2509-1.2B"

# 是否启用网络搜索（Incorrect/Ambiguous 时）；False 则用占位文案
ENABLE_WEB_SEARCH = False
