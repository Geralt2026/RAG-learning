# -*- coding: utf-8 -*-
"""
基于 MinerU 的 PDF 文本识别与切分
- 将 PDF 每页转为图像，用 MinerU 做版面分析与文本抽取
- 抽取结果按页或按块整理为 LangChain Document，供向量库与 CRAG 使用
"""
import os
from pathlib import Path

from langchain_core.documents import Document

# 延迟导入，避免未安装 MinerU 时整体报错
def _get_mineru_client():
    """获取 MinerU 客户端（懒加载，仅在需要时加载模型）。"""
    from transformers import AutoConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from mineru_vl_utils import MinerUClient
    import config as cfg
    # 先加载 config 并修补：Qwen2VLConfig 顶层无 max_position_embeddings（在 text_config 中），
    # 部分加载路径会访问 config.max_position_embeddings，导致 AttributeError，此处预先补上
    loaded_config = AutoConfig.from_pretrained(cfg.MINERU_MODEL_ID)
    if hasattr(loaded_config, "text_config") and not hasattr(loaded_config, "max_position_embeddings"):
        loaded_config.max_position_embeddings = getattr(
            loaded_config.text_config, "max_position_embeddings", 32768
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.MINERU_MODEL_ID,
        config=loaded_config,
        dtype="auto",
        device_map="auto",
    )
    # 使用 Qwen2VLProcessor 直接加载，避免 AutoProcessor 在未安装 torchvision 时走 video_processing_auto 导致 NoneType 错误
    processor = Qwen2VLProcessor.from_pretrained(
        cfg.MINERU_MODEL_ID,
        use_fast=True,
    )
    return MinerUClient(backend="transformers", model=model, processor=processor)


def _pdf_page_to_image(pdf_path: str, page_num: int):
    """将 PDF 指定页渲染为 PIL Image。使用 pymupdf 包名避免与错误的 fitz 包冲突。"""
    import pymupdf
    from PIL import Image
    import io
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    mat = pymupdf.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def extract_text_blocks_from_image(mineru_client, image):
    """
    使用 MinerU 从一页图像中抽取版面块（文本/表格等）。
    返回 ContentBlock 列表中与文本相关的内容拼接成的字符串。
    """
    extracted = mineru_client.two_step_extract(image)
    texts = []
    for block in extracted:
        # ContentBlock 有 type 和 content；仅取有文本内容的块
        content = None
        if isinstance(block, dict):
            content = block.get("content")
        else:
            content = getattr(block, "content", None)
        if content and isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return "\n\n".join(texts) if texts else ""


def load_pdfs_with_mineru(
    files_dir: str,
    mineru_client=None,
    file_ext: str = ".pdf",
) -> list[Document]:
    """
    从指定目录加载所有 PDF，使用 MinerU 做文本识别，返回 Document 列表。
    流程：PDF 每页转为图像 → MinerU 版面分析+文本抽取 → 按页整理为 LangChain Document。
    每个 Document 对应一页的抽取文本，metadata 含 source、page。
    """
    import pymupdf
    files_dir = Path(files_dir)
    if not files_dir.exists():
        return []
    # 同时匹配 .pdf / .PDF 等，避免因扩展名大小写漏掉文件
    pdf_files = list(files_dir.glob("*.pdf")) + list(files_dir.glob("*.PDF"))
    pdf_files = sorted(set(pdf_files), key=str)
    if not pdf_files:
        return []

    if mineru_client is None:
        mineru_client = _get_mineru_client()

    docs = []
    for pdf_path in sorted(pdf_files):
        pdf_path = str(pdf_path)
        try:
            doc_fitz = pymupdf.open(pdf_path)
            for page_num in range(len(doc_fitz)):
                img = _pdf_page_to_image(pdf_path, page_num)
                page_text = extract_text_blocks_from_image(mineru_client, img)
                if not page_text.strip():
                    continue
                docs.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                        },
                    )
                )
            doc_fitz.close()
        except Exception as e:
            # 单文件失败不影响其他文件，仅跳过
            docs.append(
                Document(
                    page_content="",
                    metadata={"source": os.path.basename(pdf_path), "error": str(e)},
                )
            )
    return [d for d in docs if (d.page_content and d.page_content.strip())]


def load_pdfs_with_mineru_lazy(files_dir: str, file_ext: str = ".pdf") -> list[Document]:
    """
    懒加载版：内部创建 MinerU 客户端并调用 load_pdfs_with_mineru。
    适合在「构建知识库」时调用一次。
    """
    return load_pdfs_with_mineru(files_dir, mineru_client=None, file_ext=file_ext)


def load_pdfs_with_pymupdf(files_dir: str, file_ext: str = ".pdf") -> list[Document]:
    """
    仅用 PyMuPDF 按页抽取文本，不依赖 MinerU。使用 pymupdf 包名避免与错误的 fitz 包冲突。
    """
    import pymupdf
    files_dir = Path(files_dir)
    if not files_dir.exists():
        return []
    pdf_files = list(files_dir.glob("*.pdf")) + list(files_dir.glob("*.PDF"))
    pdf_files = sorted(set(pdf_files), key=str)
    docs = []
    for pdf_path in sorted(pdf_files):
        pdf_path = str(pdf_path)
        try:
            doc_fitz = pymupdf.open(pdf_path)
            for page_num in range(len(doc_fitz)):
                page = doc_fitz[page_num]
                text = page.get_text()
                if not text.strip():
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(pdf_path), "page": page_num + 1},
                    )
                )
            doc_fitz.close()
        except Exception as e:
            docs.append(
                Document(
                    page_content="",
                    metadata={"source": os.path.basename(pdf_path), "error": str(e)},
                )
            )
    return [d for d in docs if (d.page_content and d.page_content.strip())]


if __name__ == "__main__":
    import config as cfg
    # 测试：仅当 Files 目录存在且含 PDF 时运行
    if os.path.isdir(cfg.KNOWLEDGE_FILES_DIR):
        documents = load_pdfs_with_mineru_lazy(cfg.KNOWLEDGE_FILES_DIR)
        print(f"共加载 {len(documents)} 个文档块（按页）")
        for i, d in enumerate(documents[:2]):
            print(f"[{i}] source={d.metadata.get('source')} page={d.metadata.get('page')} len={len(d.page_content)}")
    else:
        print("知识库目录不存在或为空，请创建 Experiment/CRAG/Files 并放入 PDF 后重试。")
