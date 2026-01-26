"""
基于Streamlit完成WEB网页上传服务

"""

import streamlit as st

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
uploaded_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False,  # 是否允许上传多个文件
)

if uploaded_file is not None:
    # 提取文件信息
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024 # KB

    st.subheader(f"文件信息：{file_name}")

    st.write(f"文件类型：{file_type} | 文件大小：{file_size:.2f} KB")

    # 显示文件内容 getvalue() ➡️ bytes ➡️ decode('utf-8')
    text = uploaded_file.getvalue().decode('utf-8')
    st.write(text)
    