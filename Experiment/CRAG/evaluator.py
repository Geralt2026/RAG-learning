# -*- coding: utf-8 -*-
"""
检索评估器：判断「问题-文档」相关性，输出 1.0 / -1.0
使用 PydanticAI 模型做 yes/no 判断，供 CRAG 动作触发与知识条过滤
"""
import config as cfg
from agent_model import get_evaluator_model
from pydantic_ai import Agent

EVAL_PROMPT = """请判断：下面这段文档是否包含能直接回答用户问题的信息？
只回答 yes 或 no，不要解释。

用户问题：{question}

文档内容：
{document}

回答（yes/no）："""

_eval_agent = None


def _get_eval_agent():
    global _eval_agent
    if _eval_agent is None:
        _eval_agent = Agent(get_evaluator_model(), output_type=str, output_retries=0)
    return _eval_agent


def score_relevance(question: str, document: str) -> float:
    """
    评估「问题-文档」相关性，返回 1.0（相关）或 -1.0（不相关）。
    若模型未给出有效 yes/no，则返回 0.0（视为不确定）。
    """
    document = (document or "").strip()[:4000]
    if not document:
        return -1.0
    prompt = EVAL_PROMPT.format(question=question, document=document)
    try:
        result = _get_eval_agent().run_sync(prompt)
        text = (result.output or "").strip().lower()
        first_word = (text.split() or [""])[0]
        if first_word in ("yes", "是"):
            return 1.0
        if first_word in ("no", "否"):
            return -1.0
        if text.startswith("yes") or text.startswith("是"):
            return 1.0
        if text.startswith("no") or text.startswith("否"):
            return -1.0
    except Exception:
        pass
    return 0.0
