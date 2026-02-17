# -*- coding: utf-8 -*-
"""
本地测试：扫描 Experiment/CRAG/Files 下的 PDF，用 MinerU 抽取并打印每页块数量与首块结构。
便于排查「MinerU 抽取结果为空」是识别失败还是返回结构解析问题。
"""
import os
import sys

# 保证从 CRAG 目录运行时能导入同目录模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config as cfg
from pathlib import Path

# 已知错误说明：LayerNorm shape 不匹配时给出解决建议
_MINERU_SHAPE_ERROR_MSG = """
MinerU 报错：LayerNorm weight shape 与 normalized_shape 不一致（如 1280 vs 896）。
通常原因：transformers 版本与 OpenDataLab/MinerU2.5-2509-1.2B 不兼容，或本地缓存的模型与当前库不匹配。

建议操作（按顺序尝试）：
1. 升级 transformers 到 mineru-vl-utils 推荐版本，例如：
   pip install -U "transformers>=4.56.0"
2. 删除本地模型缓存后重新运行（会重新下载）：
   删除目录：%USERPROFILE%\\.cache\\modelscope\\hub\\models\\OpenDataLab\\MinerU2.5-2509-1.2B
3. 若使用 ModelScope，可指定从 HuggingFace 拉取（或反之）：
   设置环境变量 MINERU_MODEL_SOURCE=huggingface 或 modelscope 后重试
4. 查看 mineru-vl-utils 的 GitHub Issues 是否有同款报错及官方修复
"""


def main():
    files_dir = Path(cfg.KNOWLEDGE_FILES_DIR)
    if not files_dir.exists():
        print(f"目录不存在: {files_dir}")
        return
    pdf_files = list(files_dir.glob("*.pdf")) + list(files_dir.glob("*.PDF"))
    pdf_files = sorted(set(pdf_files), key=str)
    print(f"找到 {len(pdf_files)} 个 PDF: {[p.name for p in pdf_files]}\n")

    from pdf_loader import _get_mineru_client, _pdf_page_to_image, extract_blocks_from_image, _block_to_item

    client = _get_mineru_client()
    for pdf_path in pdf_files:
        pdf_path = str(pdf_path)
        import pymupdf
        doc = pymupdf.open(pdf_path)
        print(f"--- {os.path.basename(pdf_path)} (共 {len(doc)} 页) ---")
        for page_num in range(min(3, len(doc))):  # 只测前 3 页
            img = _pdf_page_to_image(pdf_path, page_num)
            extracted = client.two_step_extract(img)
            print(f"  页 {page_num + 1}: two_step_extract 返回 {len(extracted)} 个 block, type={type(extracted)}")
            if extracted:
                first = extracted[0]
                print(f"    首 block type={type(first)}, __dict__={getattr(first, '__dict__', first)}")
                if hasattr(first, "content"):
                    print(f"    first.content = {repr(getattr(first, 'content', None))[:200]}")
            # 无论版面是否检出块，都走 extract_blocks_from_image（空时会整页 OCR 回退）
            blocks = extract_blocks_from_image(client, img)
            print(f"    解析后有效 block 数: {len(blocks)}")
        doc.close()
        print()
    print("完成。")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "normalized_shape" in str(e) and "weight" in str(e):
            print(_MINERU_SHAPE_ERROR_MSG)
        print(f"RuntimeError: {e}")
        sys.exit(1)