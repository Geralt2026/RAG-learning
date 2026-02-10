# -*- coding: utf-8 -*-
"""
Streamlit 交互界面：CRAG 知识库问答
"""
# Workaround: huggingface_hub 1.4.x 懒加载在某些导入顺序下会报 cannot import 'is_offline_mode'
# 在任意包执行 from huggingface_hub import is_offline_mode 之前，先把 is_offline_mode 注入到主模块
try:
    import huggingface_hub.constants as _hf_constants
    import huggingface_hub as _hf
    if not hasattr(_hf, "is_offline_mode"):
        _hf.is_offline_mode = _hf_constants.is_offline_mode
except Exception:
    pass

import streamlit as st
from crag import run_crag
from knowledge_base import build_knowledge_base

st.set_page_config(page_title="CRAG 知识库问答", page_icon="📚", layout="centered")
st.title("📚 CRAG 知识库问答")
st.caption("基于 Corrective RAG：先评估检索质量，再决定使用精炼知识或外部补充")

# 侧边栏：知识库重建
with st.sidebar:
    st.subheader("知识库管理")
    if st.button("🔄 重新构建知识库", help="从 Experiment/CRAG/Files 下的 PDF 重新抽取并建库（需 MinerU）"):
        with st.spinner("正在使用 MinerU 处理 PDF 并写入向量库…"):
            msg = build_knowledge_base(force_rebuild=True)
        st.success(msg)

# 主区域：问答
question = st.chat_input("输入您的问题…")
if question:
    with st.spinner("检索与生成中…"):
        result = run_crag(question)
    action = result["action"]
    action_cn = {"correct": "✅ 使用知识库（精炼）", "incorrect": "⚠️ 使用外部说明", "ambiguous": "🔀 知识库+外部补充"}.get(action, action)
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"动作：{action_cn} | 检索到 {result.get('num_retrieved', 0)} 段文档")
    with st.expander("查看使用的参考资料"):
        st.text(result.get("context_used", "（无）"))
