import streamlit as st

def custom_css():
    """添加自定义CSS样式"""
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b9bff;
        color: white;
    }
    .agent-selector {
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
        background-color: #f7f7f7;
    }
    .chat-container {
        border-radius: 10px;
        background-color: white;
        padding: 10px;
        height: calc(100vh - 250px);
        overflow-y: auto;
        margin-bottom: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .debug-container {
        border-radius: 10px;
        background-color: white;
        height: calc(100vh - 120px);
        overflow-y: auto;
        padding: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .example-question {
        background-color: #f7f7f7;
        padding: 8px;
        border-radius: 4px;
        margin: 5px 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-question:hover {
        background-color: #e6e6e6;
    }
    .settings-bar {
        padding: 10px;
        background-color: #f7f7f7;
        border-radius: 5px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    /* 源内容样式 - 改进版 */
    .source-content-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #e0e0e0;
    }
    .source-content {
        white-space: pre-wrap;
        word-wrap: break-word;
        background-color: #f5f5f5;
        padding: 16px;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        color: #24292e;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
    }
    /* 调试信息样式 */
    .debug-header {
        background-color: #eef2f5;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #4b9bff;
    }
    /* 知识图谱控制面板 */
    .kg-controls {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e6e6e6;
    }
    /* 按钮悬停效果 */
    button:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    /* 源内容按钮样式 */
    .view-source-button {
        background-color: #f1f8ff;
        border: 1px solid #c8e1ff;
        color: #0366d6;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 12px;
        margin: 4px;
    }
    .view-source-button:hover {
        background-color: #dbedff;
    }
    /* 反馈按钮样式 */
    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }
    .feedback-positive {
        color: #0F9D58;
        font-weight: bold;
    }
    .feedback-negative {
        color: #DB4437;
        font-weight: bold;
    }
    .feedback-given {
        opacity: 0.7;
        font-style: italic;
    }
    /* 操作中状态提示 */
    .processing-indicator {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 4px;
        border-left: 4px solid #ffeeba;
        margin: 5px 0;
        font-size: 12px;
    }
    /* 迭代轮次样式 */
    .iteration-round {
        background-color: #f8f9fa;
        border-left: 4px solid #4285F4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    .iteration-query {
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: monospace;
        margin: 5px 0;
    }
    
    .iteration-info {
        background-color: #e8f5e9;
        padding: 12px;
        border-radius: 4px;
        border-left: 3px solid #4CAF50;
        margin: 10px 0;
    }
    
    /* 进度条样式 */
    .iteration-progress {
        height: 8px;
        background-color: #f0f0f0;
        border-radius: 4px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .iteration-progress-bar {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

KG_MANAGEMENT_CSS = """
<style>
    .kg-form {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .kg-form-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #1976d2;
    }
    .kg-entity-card {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .kg-relation-card {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .kg-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 8px;
    }
    .kg-entity-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .kg-relation-badge {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    .kg-property-table {
        width: 100%;
        border-collapse: collapse;
    }
    .kg-property-table th, .kg-property-table td {
        border: 1px solid #e0e0e0;
        padding: 8px;
        text-align: left;
    }
    .kg-property-table th {
        background-color: #f5f5f5;
    }
</style>
"""