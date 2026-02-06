# -*- coding: utf-8 -*-
"""
外部知识获取：Incorrect / Ambiguous 时使用
查询改写使用 PydanticAI；未启用网络搜索时返回占位文案
"""
import config as cfg
from agent_model import get_generator_agent
from refinement import refine_knowledge


def rewrite_query_for_search(question: str) -> str:
    """将用户问题改写成 2～3 个关键词，便于搜索引擎查询。"""
    prompt = f"""请从下面问题中提取 2～3 个关键词，用逗号分隔，用于网络搜索。只输出关键词，不要解释。
问题：{question}
关键词："""
    try:
        result = get_generator_agent().run_sync(prompt)
        text = (result.output or "").strip()
        return text or question
    except Exception:
        return question


def web_search_or_placeholder(question: str) -> str:
    """
    当检索被判为 Incorrect 或需补充时，获取外部知识。
    若未配置真实搜索 API，返回占位说明。
    """
    if not cfg.ENABLE_WEB_SEARCH:
        return "（当前未启用网络搜索。知识库中未找到与您问题高度相关的内容，请尝试换一种问法或补充更多背景。）"
    query = rewrite_query_for_search(question)
    return "（网络搜索功能已启用但尚未配置 API，请稍后使用或联系管理员。）"
