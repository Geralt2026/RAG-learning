import time
from rag import RAGService
import config_data as config
import streamlit as st

# 添加网页标题
st.title("知识库Agent")
st.divider()  # 分隔符

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是知识库Agent，有什么可以帮你的吗？"}]

if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:

    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt}) 

    ai_res_list = []
    with st.spinner("Thinking..."):
        result = st.session_state["rag_service"].chain.stream({"input": prompt}, config=config.session_config)
        
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(result, ai_res_list))
        st.session_state["messages"].append({"role": "assistant", "content": "".join(ai_res_list)})
