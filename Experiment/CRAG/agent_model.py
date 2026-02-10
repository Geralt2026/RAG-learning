# -*- coding: utf-8 -*-
"""
PydanticAI 模型与 Agent：统一使用 qwen3-max（通义），供 CRAG 评估与生成
"""
from openai import AsyncOpenAI

import config as cfg
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_openai_client():
    """通义 / OpenAI 兼容的 AsyncOpenAI 客户端。"""
    base = cfg.OPENAI_API_BASE or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return AsyncOpenAI(
        base_url=base,
        api_key=cfg.OPENAI_API_KEY or None,
    )


def get_model():
    """返回 PydanticAI 使用的聊天模型（qwen3-max）。"""
    client = get_openai_client()
    return OpenAIChatModel(
        cfg.LLM_MODEL,
        provider=OpenAIProvider(openai_client=client),
    )


# 生成用 Agent：基于参考资料回答用户问题，输出纯文本
_generator_model = None


def get_generator_agent():
    """获取用于最终回答生成的 PydanticAI Agent（output_type=str）。"""
    global _generator_model
    if _generator_model is None:
        _generator_model = get_model()
    return Agent(
        _generator_model,
        output_type=str,
        output_retries=0,
    )


# 评估用 Agent：仅做 yes/no 判断，同步调用
_evaluator_model = None


def get_evaluator_model():
    """获取用于相关性评估的模型（单轮 yes/no）。"""
    global _evaluator_model
    if _evaluator_model is None:
        _evaluator_model = get_model()
    return _evaluator_model
