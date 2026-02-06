# -*- coding: utf-8 -*-
"""
知识精炼（Decompose-then-Recompose）：
将检索到的文档切分为「条」→ 对每条做相关性打分 → 过滤低分条 → 按顺序重组
"""
import re
import config as cfg
from evaluator import score_relevance


def split_into_strips(text: str, max_chars: int = 400) -> list[str]:
    """
    将长文本分解为若干「条」（strip），每条约 max_chars 字符，尽量按句边界切。
    """
    if not (text or text.strip()):
        return []
    text = text.strip()
    # 先按句号、问号、换行等分句
    sentences = re.split(r'(?<=[。！？\n])\s*', text)
    strips = []
    current = []
    current_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if current_len + len(s) > max_chars and current:
            strips.append("".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += len(s)
    if current:
        strips.append("".join(current))
    return strips


def refine_knowledge(
    question: str,
    documents: list,
    score_fn=None,
    strip_top_k: int = None,
    strip_threshold: float = None,
) -> str:
    """
    知识精炼（Decompose-then-Recompose）：对多段文档分解为条 → 每条打分 → 过滤低分条 → 按顺序重组。
    documents 可为 list[Document] 或 list[str]（page_content 或直接字符串）。
    score_fn(question, doc_text) -> float，默认使用 evaluator.score_relevance。
    """
    if score_fn is None:
        score_fn = score_relevance
    strip_top_k = strip_top_k or cfg.STRIP_TOP_K
    strip_threshold = strip_threshold or cfg.STRIP_FILTER_THRESHOLD

    all_strips_with_scores = []
    for doc in documents:
        content = getattr(doc, "page_content", None) or (doc if isinstance(doc, str) else "")
        if not (content and content.strip()):
            continue
        strips = split_into_strips(content)
        for strip in strips:
            if not strip.strip():
                continue
            score = score_fn(question, strip)
            all_strips_with_scores.append((strip, score))

    # 过滤：保留 score > strip_threshold 的条，再按分数取 top_k（保留顺序则按出现顺序取前 k 个高分的）
    filtered = [(s, sc) for s, sc in all_strips_with_scores if sc > strip_threshold]
    if not filtered:
        # 若全部低于阈值，则按分数排序取 top_k
        filtered = sorted(all_strips_with_scores, key=lambda x: -x[1])[:strip_top_k]
    else:
        # 按原顺序保留，但只取前 strip_top_k 条（或全部）
        filtered = filtered[:strip_top_k] if len(filtered) > strip_top_k else filtered

    return "\n\n".join(s for s, _ in filtered).strip() if filtered else ""
