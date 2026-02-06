# -*- coding: utf-8 -*-
"""
CRAG 主流程（PydanticAI）：检索 → 评估 → 动作触发 → 知识准备 → 生成
"""
import config as cfg
from action_trigger import CORRECT, INCORRECT, AMBIGUOUS, trigger_action
from agent_model import get_generator_agent
from evaluator import score_relevance
from refinement import refine_knowledge
from search import web_search_or_placeholder
from vector_store import get_retriever

GENERATE_PROMPT = """你是一个基于知识库的问答助手。请严格依据下面提供的「参考资料」回答用户问题；若资料中未包含答案，请明确说明「根据现有资料无法回答」并建议用户换一种问法或补充信息。

参考资料：
{context}

用户问题：{question}

请直接给出回答（不要复述问题）："""


def run_crag(question: str, retriever=None) -> dict:
    """
    执行 CRAG 流程，返回 answer、action、context_used 等。
    使用 PydanticAI Agent 进行相关性评估与最终生成。
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入有效问题。", "action": INCORRECT, "context_used": "", "scores": [], "num_retrieved": 0}

    retriever = retriever or get_retriever()

    # 1) 检索
    docs = retriever.invoke(question)
    if not docs:
        docs = []

    # 2) 评估：PydanticAI 模型对每段文档打相关性分数
    scores = []
    for d in docs:
        content = getattr(d, "page_content", None) or ""
        scores.append(score_relevance(question, content))

    # 3) 动作触发
    action = trigger_action(scores)

    # 4) 知识准备
    internal_knowledge = ""
    external_knowledge = ""
    if action == CORRECT:
        internal_knowledge = refine_knowledge(question, docs)
    elif action == INCORRECT:
        external_knowledge = web_search_or_placeholder(question)
    else:
        internal_knowledge = refine_knowledge(question, docs)
        external_knowledge = web_search_or_placeholder(question)

    context_parts = []
    if internal_knowledge and internal_knowledge.strip():
        context_parts.append("【来自知识库】\n" + internal_knowledge)
    if external_knowledge and external_knowledge.strip():
        context_parts.append("【补充说明】\n" + external_knowledge)
    context_used = "\n\n".join(context_parts).strip() or "（无可用参考资料）"

    # 5) 生成：PydanticAI Agent
    prompt = GENERATE_PROMPT.format(context=context_used, question=question)
    try:
        result = get_generator_agent().run_sync(prompt)
        answer = (result.output or "").strip()
    except Exception as e:
        answer = f"生成回答时出错：{e}"

    return {
        "answer": answer,
        "action": action,
        "context_used": context_used,
        "scores": scores,
        "num_retrieved": len(docs),
    }
