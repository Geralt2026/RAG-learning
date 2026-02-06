# -*- coding: utf-8 -*-
"""
动作触发：根据检索文档的相关性分数，决定 Correct / Incorrect / Ambiguous
"""
import config as cfg

# 动作枚举，与论文一致
CORRECT = "correct"      # 至少有一份文档相关性高，使用精炼后的内部知识
INCORRECT = "incorrect"  # 全部文档相关性低，弃用检索，改用外部知识（网络搜索或占位）
AMBIGUOUS = "ambiguous"  # 介于两者之间，同时使用内部精炼知识 + 外部知识


def trigger_action(
    scores: list[float],
    upper: float = None,
    lower: float = None,
) -> str:
    """
    根据各文档的 score 列表判断触发动作。
    - 任一 score > upper -> Correct
    - 全部 score < lower -> Incorrect
    - 否则 -> Ambiguous
    """
    upper = upper if upper is not None else cfg.THRESHOLD_UPPER
    lower = lower if lower is not None else cfg.THRESHOLD_LOWER
    if not scores:
        return INCORRECT
    if any(s > upper for s in scores):
        return CORRECT
    if all(s < lower for s in scores):
        return INCORRECT
    return AMBIGUOUS
