# -*- coding: utf-8 -*-
"""
基于 MinerU 的 PDF 图像文本识别与按多级标题切分
- 取文字全部交给 MinerU：PDF 先整份转成图片，再统一交给 MinerU 批量识别
- PyMuPDF 仅用于「PDF→图片」渲染，不参与任何文字抽取
- 识别多级标题（1、1.1、1.1.1 等）并按标题拆分为独立文档块，便于检索
"""
import os
import re
from pathlib import Path

from langchain_core.documents import Document


def _get_mineru_client():
    """获取 MinerU 客户端（懒加载）。"""
    from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient
    import config as cfg
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.MINERU_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    # 新版本 transformers 中 Qwen2VLConfig 的 max_position_embeddings 在 text_config 里，
    # mineru_vl_utils 访问的是 model.config.max_position_embeddings，需补到顶层
    if not hasattr(model.config, "max_position_embeddings") and hasattr(
        model.config, "text_config"
    ):
        model.config.max_position_embeddings = getattr(
            model.config.text_config, "max_position_embeddings", 32768
        )
    processor = AutoProcessor.from_pretrained(
        cfg.MINERU_MODEL_ID,
        use_fast=True,
    )
    return MinerUClient(backend="transformers", model=model, processor=processor)


def _pdf_page_to_image(pdf_path: str, page_num: int):
    """将 PDF 指定页渲染为 PIL Image（仅渲染，不取文字；文字全部由 MinerU 识别）。"""
    import pymupdf
    from PIL import Image
    import io
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    mat = pymupdf.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def _pdf_to_images(pdf_path: str) -> list:
    """
    将 PDF 全部页渲染为 PIL Image 列表。仅做「PDF→图片」转换，不抽取文字。
    文字识别统一交给 MinerU 处理。
    """
    import pymupdf
    from PIL import Image
    import io
    doc = pymupdf.open(pdf_path)
    images = []
    mat = pymupdf.Matrix(1.5, 1.5)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def _block_to_item(block) -> dict | None:
    """从 MinerU 块中取出 type 与 content，兼容 dict/对象。"""
    if block is None:
        return None
    typ = None
    content = None
    if isinstance(block, dict):
        typ = block.get("type")
        content = block.get("content") or block.get("text") or block.get("data")
    else:
        typ = getattr(block, "type", None)
        content = getattr(block, "content", None) or getattr(block, "text", None)
    if content is None and hasattr(block, "__dict__"):
        d = getattr(block, "__dict__", {})
        content = d.get("content") or d.get("text")
        if typ is None:
            typ = d.get("type")
    if content and isinstance(content, str) and content.strip():
        return {"type": typ or "text", "content": content.strip()}
    return None


def extract_blocks_from_image(mineru_client, image, fallback_full_page: bool = True) -> list[dict]:
    """
    使用 MinerU 从一页图像中抽取版面块（含 type 与 content）。
    返回 [{"type": "text"|"table"|..., "content": "..."}, ...]。
    若版面检测返回 0 块或所有块均无文本（如仅 image/list），且 fallback_full_page 为 True，
    则对整页做一次文本识别并返回单块，避免「共加载 0 个文档块」。
    """
    extracted = mineru_client.two_step_extract(image)
    blocks = []
    for block in extracted:
        item = _block_to_item(block)
        if item:
            blocks.append(item)
    if not blocks and fallback_full_page:
        # Layout 返回 0 块或全部无 content 时，整页当作文本识别
        full_text = mineru_client.content_extract(image, type="text")
        if full_text and full_text.strip():
            blocks = [{"type": "text", "content": full_text.strip()}]
            if os.environ.get("CRAG_MINERU_DEBUG"):
                print("  [MinerU] 本页版面未检出块，已用整页文本识别回退。")
    return blocks


# 多级标题行正则：匹配行首的 1 / 1.1 / 1.1.1 / 2.3.4 等编号（可跟点或空格，后接标题文字）
_HEADING_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)[\.\s]+\s*(.*)$",
    re.MULTILINE,
)


def _heading_level(numbering: str) -> int:
    """根据编号计算层级：1 -> 1, 1.1 -> 2, 1.1.1 -> 3。"""
    return numbering.count(".") + 1


def split_by_headings(full_text: str) -> list[tuple[int, str, str]]:
    """
    按多级标题切分全文。返回 [(level, heading_line, section_text), ...]。
    heading_line 含编号+标题；section_text 为该节从标题行到下一同级/更高级标题之前的内容。
    支持 1 / 1.1 / 1.1.1 等格式。
    """
    if not full_text or not full_text.strip():
        return []
    lines = full_text.split("\n")
    sections = []
    current_heading_level = None
    current_heading_line = None
    current_lines = []
    preamble_lines = []  # 第一个标题之前的内容

    def flush_section():
        if current_heading_line is None:
            return
        body = "\n".join(current_lines).strip()
        section_text = (current_heading_line + "\n" + body) if body else current_heading_line
        if section_text.strip():
            sections.append((current_heading_level, current_heading_line.strip(), section_text.strip()))

    for line in lines:
        m = _HEADING_PATTERN.match(line)
        if m:
            num_part = m.group(1)
            rest = m.group(2).strip()
            level = _heading_level(num_part)
            # 视为标题：编号后内容较短，避免正文中带编号的长句被当标题
            is_heading = len(rest) < 200 and len(line) < 250
            if is_heading:
                # 若此前有前言内容，先作为一节输出
                if current_heading_line is None and preamble_lines:
                    preamble_text = "\n".join(preamble_lines).strip()
                    if preamble_text:
                        sections.append((0, "(前言或未编号)", preamble_text))
                    preamble_lines = []
                flush_section()
                current_heading_level = level
                current_heading_line = line
                current_lines = []
                continue
        # 非标题行：归入当前节或前言
        if current_heading_line is not None:
            current_lines.append(line)
        else:
            preamble_lines.append(line)

    flush_section()
    if preamble_lines and not sections:
        preamble_text = "\n".join(preamble_lines).strip()
        if preamble_text:
            sections.append((0, "(全文)", preamble_text))
    return sections


def _blocks_to_page_text(mineru_client, page_blocks: list, fallback_image=None) -> str:
    """将一页的 MinerU 块转为该页文本；若无有效块且提供 fallback_image 则整页 OCR 回退。"""
    parts = []
    for block in page_blocks:
        item = _block_to_item(block)
        if item and item.get("content"):
            parts.append(item["content"])
    if not parts and fallback_image is not None:
        full_text = mineru_client.content_extract(fallback_image, type="text")
        if full_text and full_text.strip():
            return full_text.strip()
    return "\n\n".join(parts) if parts else ""


def load_pdfs_with_mineru(
    files_dir: str,
    mineru_client=None,
    file_ext: str = ".pdf",
) -> list[Document]:
    """
    取文字全部交给 MinerU：PDF 先整份转成图片，再统一交给 MinerU 批量识别。
    不使用 PyMuPDF 抽文字（仅用其做 PDF→图片 渲染）；某页无结果时用整页 OCR 回退。
    """
    files_dir = Path(files_dir)
    if not files_dir.exists():
        return []
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
            # 1. PDF → 全部页图片（仅渲染，不取文字）
            images = _pdf_to_images(pdf_path)
            if not images:
                continue
            # 2. 统一丢给 MinerU 批量处理
            blocks_per_page = mineru_client.batch_two_step_extract(images)
            # 3. 按页拼文本，无块时用整页 OCR 回退
            full_parts = []
            for i, page_blocks in enumerate(blocks_per_page):
                fallback_img = images[i] if i < len(images) else None
                page_text = _blocks_to_page_text(mineru_client, page_blocks, fallback_img)
                if page_text:
                    full_parts.append(page_text)
            full_text = "\n\n".join(full_parts)
            if not full_text.strip():
                continue

            # 按多级标题切分
            sections = split_by_headings(full_text)
            if not sections:
                # 无标题结构则整篇作为一个块
                docs.append(
                    Document(
                        page_content=full_text.strip(),
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "section": "全文",
                            "heading_level": 0,
                        },
                    )
                )
            else:
                for level, heading_line, section_text in sections:
                    if not section_text.strip():
                        continue
                    docs.append(
                        Document(
                            page_content=section_text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "section": heading_line[:200],
                                "heading_level": level,
                            },
                        )
                    )
        except Exception as e:
            docs.append(
                Document(
                    page_content="",
                    metadata={"source": os.path.basename(pdf_path), "error": str(e)},
                )
            )
    return [d for d in docs if (d.page_content and d.page_content.strip())]


def load_pdfs_with_mineru_lazy(files_dir: str, file_ext: str = ".pdf") -> list[Document]:
    """懒加载版：内部创建 MinerU 客户端并调用 load_pdfs_with_mineru。"""
    return load_pdfs_with_mineru(files_dir, mineru_client=None, file_ext=file_ext)


def load_pdfs_with_pymupdf(files_dir: str, file_ext: str = ".pdf") -> list[Document]:
    """
    仅用 PyMuPDF 按页抽取文本（不经过 MinerU）。供需要纯文字版 PDF 且不用 MinerU 的场景。
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
    if os.path.isdir(cfg.KNOWLEDGE_FILES_DIR):
        documents = load_pdfs_with_mineru_lazy(cfg.KNOWLEDGE_FILES_DIR)
        print(f"共加载 {len(documents)} 个文档块（按多级标题切分）")
        for i, d in enumerate(documents[:5]):
            print(f"[{i}] source={d.metadata.get('source')} section={d.metadata.get('section')} level={d.metadata.get('heading_level')} len={len(d.page_content)}")
    else:
        print("知识库目录不存在或为空，请创建 Experiment/CRAG/Files 并放入 PDF 后重试。")
