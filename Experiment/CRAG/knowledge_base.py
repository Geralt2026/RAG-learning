# -*- coding: utf-8 -*-
"""
知识库构建：强制使用 MinerU 从 Experiment/CRAG/Files 下的 PDF 做版面分析与文本抽取，切分后写入向量库
"""
import os
from pathlib import Path

import config as cfg
from pdf_loader import load_pdfs_with_mineru_lazy
from vector_store import get_vector_store, get_splitter


def _count_pdf_files(files_dir: str) -> int:
    """统计目录下 PDF 文件数量（.pdf/.PDF 去重，Windows 下同一文件只算一个）。"""
    p = Path(files_dir)
    if not p.is_dir():
        return 0
    all_pdf = list(p.glob("*.pdf")) + list(p.glob("*.PDF"))
    return len(set(all_pdf))


def build_knowledge_base(force_rebuild: bool = False) -> str:
    """
    使用 MinerU 从 Files 目录加载 PDF（版面分析+文本识别），切分后写入 Chroma。
    force_rebuild=True 时先清空再建。
    """
    if not os.path.isdir(cfg.KNOWLEDGE_FILES_DIR):
        return f"知识库目录不存在：{cfg.KNOWLEDGE_FILES_DIR}，请创建 Experiment/CRAG/Files 并放入 PDF 后重试。"

    num_pdf = _count_pdf_files(cfg.KNOWLEDGE_FILES_DIR)
    if num_pdf == 0:
        return f"知识库目录下未找到 PDF 文件（.pdf/.PDF）。当前目录：{os.path.abspath(cfg.KNOWLEDGE_FILES_DIR)}"

    # 强制使用 MinerU 做 PDF 识别
    documents = load_pdfs_with_mineru_lazy(cfg.KNOWLEDGE_FILES_DIR)
    if not documents:
        return f"找到 {num_pdf} 个 PDF 文件，但 MinerU 抽取结果为空（可能为扫描版或识别失败）。目录：{os.path.abspath(cfg.KNOWLEDGE_FILES_DIR)}"

    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    if not chunks:
        return "切分后无有效文本块。"

    if force_rebuild:
        import shutil
        if os.path.exists(cfg.CHROMA_PERSIST_DIR):
            shutil.rmtree(cfg.CHROMA_PERSIST_DIR)
        os.makedirs(cfg.CHROMA_PERSIST_DIR, exist_ok=True)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return f"成功（MinerU）：共处理 {len(documents)} 页，切分为 {len(chunks)} 块并已写入向量库。"
