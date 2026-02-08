# -*- coding: utf-8 -*-
import streamlit as st
from crag import run_crag
from knowledge_base import build_knowledge_base

st.set_page_config(page_title="CRAG 知识库问答", page_icon="📚", layout="centered")
st.title("📚 CRAG 知识库问答")
st.caption("基于 Corrective RAG：先评估检索质量，再决定使用精炼知识或外部补充")

# 用 session_state 保存历史对话，避免刷新后丢失
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏：知识库重建
with st.sidebar:
    st.subheader("知识库管理")
    if st.button("🔄 重新构建知识库", help="从 Experiment/CRAG/Files 下的 PDF 重新抽取并建库（需 MinerU）"):
        with st.spinner("正在使用 MinerU 处理 PDF 并写入向量库…"):
            msg = build_knowledge_base(force_rebuild=True)
        st.success(msg)

# 主区域：先渲染历史对话，再处理新输入
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            st.caption(msg.get("caption", ""))
            if msg.get("context_used") is not None:
                with st.expander("查看使用的参考资料"):
                    st.text(msg["context_used"])

question = st.chat_input("输入您的问题…")
if question:
    # 把用户问题加入历史并立刻展示
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("检索与生成中…"):
        result = run_crag(question)
    action = result["action"]
    action_cn = {"correct": "✅ 使用知识库（精炼）", "incorrect": "⚠️ 使用外部说明", "ambiguous": "🔀 知识库+外部补充"}.get(action, action)
    caption = f"动作：{action_cn} | 检索到 {result.get('num_retrieved', 0)} 段文档"
    # 把助手回答加入历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "caption": caption,
        "context_used": result.get("context_used", "（无）"),
    })
    st.rerun()
