# -*- coding: utf-8 -*-
"""
CRAG 向量库：基于 Chroma，支持 Ollama / OpenAI 等 Embedding
"""
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as cfg


def get_embedding_model():
    """根据配置返回 LangChain Embedding 实例（检索用；LLM 已改用 PydanticAI）。"""
    if cfg.EMBEDDING_PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=cfg.OLLAMA_EMBED_MODEL)
    if cfg.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=cfg.OPENAI_EMBED_MODEL,
            api_key=cfg.OPENAI_API_KEY or None,
            base_url=cfg.OPENAI_API_BASE or None,
        )
    if cfg.EMBEDDING_PROVIDER == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model=cfg.OPENAI_EMBED_MODEL)
    raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {cfg.EMBEDDING_PROVIDER}")


def get_vector_store():
    """创建或加载 Chroma 向量库。"""
    from langchain_chroma import Chroma
    os.makedirs(cfg.CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name=cfg.CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=cfg.CHROMA_PERSIST_DIR,
    )


def get_splitter():
    """文本切分器：将长文档切成块，便于检索。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "？", "！", " ", ""],
        length_function=len,
    )


def get_retriever(k: int = None):
    """获取检索器，k 为返回文档数，默认使用配置中的 RETRIEVE_TOP_K。"""
    vs = get_vector_store()
    return vs.as_retriever(search_kwargs={"k": k or cfg.RETRIEVE_TOP_K})
