# LangChain RAG 项目技术文档

## 📋 项目概述

本项目是一个基于 LangChain 框架构建的 RAG（Retrieval-Augmented Generation，检索增强生成）系统，集成了向量数据库、对话历史管理和 Web 界面，实现了智能问答和知识库管理功能。

## 🏗️ 项目结构

```
LangChain_RAG_Proj/
├── app_qa.py                 # Streamlit 问答界面
├── app_file_uploader.py     # Streamlit 文件上传界面
├── rag.py                    # RAG 服务核心模块
├── knowledge_base.py         # 知识库管理模块
├── vector_stores.py          # 向量存储服务模块
├── file_history_store.py     # 文件历史记录存储模块
├── config_data.py            # 项目配置文件
├── data/                     # 数据目录
│   ├── chroma_db/           # Chroma 向量数据库存储
│   ├── md5.txt              # MD5 校验文件（防重复上传）
│   └── *.txt                # 知识库文本文件
└── chat_history/            # 对话历史存储目录
    └── user_001             # 用户会话历史文件
```

## 🛠️ 技术栈

- **LangChain**: 大语言模型应用开发框架
- **Chroma**: 开源向量数据库
- **DashScope Embeddings**: 阿里云百炼嵌入模型（text-embedding-v4）
- **ChatTongyi**: 阿里云通义千问大模型（qwen3-max）
- **Streamlit**: Web 应用框架
- **Python 3.10+**: 开发语言

## 📁 核心模块详解

### 1. 配置文件 (`config_data.py`)

**位置**: `LangChain_RAG_Proj/config_data.py`

**作用**: 集中管理项目所有配置参数

**配置项说明**:

```python
# 文件路径配置
md5_path = "LangChain_RAG_Proj\data\md5.txt"  # MD5 校验文件路径

# Chroma 向量数据库配置
chroma_collection_name = "rag"  # 向量库集合名称
chroma_persist_directory = "LangChain_RAG_Proj\data\chroma_db"  # 持久化存储路径

# 文本切分配置
chunk_size = 1000  # 每个文本块的最大字符数
chunk_overlap = 50  # 文本块之间的重叠字符数
separators = ["\n\n", "\n", "。", "，", "？", "！", "：", "；", "、", "|", " "]  # 文本分隔符
max_split_char_number = 1000  # 文本长度超过此值才进行切分

# 检索配置
similarity_threshold = 1  # 相似度检索返回的文档数量（k值）

# 模型配置
embedding_model = "text-embedding-v4"  # 嵌入模型名称
chat_model = "qwen3-max"  # 对话模型名称

# 会话配置
session_config = {"configurable": {"session_id": "user_001"}}  # 默认会话ID
```

**使用方式**: 所有模块通过 `import config_data as config` 导入配置

### 2. 知识库管理模块 (`knowledge_base.py`)

**核心功能**:
- 文档向量化和存储
- MD5 去重机制
- 文本智能切分

**技术原理**:

1. **MD5 去重机制**:
   ```python
   def get_string_md5(input_str: str) -> str:
       """计算字符串的 MD5 值"""
       str_bytes = input_str.encode("utf-8")
       md5 = hashlib.md5()
       md5.update(str_bytes)
       return md5.hexdigest()
   ```
   - 对上传的文本内容计算 MD5 哈希值
   - 在 `md5.txt` 文件中记录已处理的 MD5 值
   - 上传前检查 MD5，避免重复处理相同内容

2. **文本切分策略**:
   ```python
   RecursiveCharacterTextSplitter(
       chunk_size=1000,        # 每个块最大 1000 字符
       chunk_overlap=50,       # 块之间重叠 50 字符（保持上下文）
       separators=["\n\n", "\n", "。", ...]  # 按优先级尝试分割
   )
   ```
   - 优先按段落（`\n\n`）分割
   - 其次按换行符（`\n`）分割
   - 再次按中文标点符号分割
   - 重叠设计确保上下文连贯性

3. **向量化存储流程**:
   ```
   文本内容 → MD5 校验 → 文本切分 → 生成 Embedding → 存入 Chroma → 记录 MD5
   ```

### 3. 向量存储服务 (`vector_stores.py`)

**核心功能**: 封装 Chroma 向量数据库操作

**技术原理**:

```python
class VectorStoreService:
    def __init__(self, embedding):
        self.vector_store = Chroma(
            collection_name="rag",
            embedding_function=embedding,  # DashScope Embeddings
            persist_directory=".../chroma_db"  # 持久化路径
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": 1}  # 返回最相似的 1 个文档
        )
```

**检索原理**:
1. 用户查询 → Embedding 向量化
2. 在向量库中计算余弦相似度
3. 返回 top-k 个最相似的文档块

### 4. 对话历史存储 (`file_history_store.py`)

**核心功能**: 基于文件的对话历史持久化存储

**技术原理**:

1. **文件存储结构**:
   ```
   chat_history/
   └── user_001  # 以 session_id 为文件名
   ```
   - 每个会话 ID 对应一个独立的 JSON 文件
   - 文件内容为消息列表的 JSON 序列化

2. **消息序列化**:
   ```python
   # 存储：BaseMessage → dict
   message_to_dict(message)  # LangChain 提供的序列化方法
   
   # 读取：dict → BaseMessage
   messages_from_dict(messages_data)  # LangChain 提供的反序列化方法
   ```

3. **历史记录管理**:
   ```python
   class FileChatMessageHistory(BaseChatMessageHistory):
       def add_messages(self, messages):
           # 读取已有消息 + 新消息 → 合并 → 写入文件
           all_messages = list(self.messages)
           all_messages.extend(messages)
           # 序列化并保存
       
       @property
       def messages(self):
           # 从文件读取 → 反序列化 → 返回 BaseMessage 列表
   ```

### 5. RAG 服务核心 (`rag.py`)

**核心功能**: 实现检索增强生成流程

**技术架构**:

```
用户输入
    ↓
[RunnableWithMessageHistory]  # 自动注入历史记录
    ↓
{
    "input": RunnablePassthrough(),  # 传递用户输入
    "context": format_for_retriever  # 提取查询文本
        → retriever.invoke()         # 向量检索
        → format_docs()              # 格式化文档
}
    ↓
format_for_prompt_template  # 重组数据：{input, context, history}
    ↓
ChatPromptTemplate  # 构建提示词
    ↓
ChatTongyi  # 调用大模型
    ↓
StrOutputParser  # 解析输出
    ↓
返回答案
```

**关键技术点**:

1. **RunnableWithMessageHistory 工作原理**:
   ```python
   conversation_chain = RunnableWithMessageHistory(
       chain,                    # 基础链
       get_history,              # 历史记录获取函数
       input_messages_key="input",      # 输入键名
       history_messages_key="history"   # 历史记录键名
   )
   ```
   - 自动调用 `get_history(session_id)` 获取历史记录
   - 将历史记录注入到 prompt 的 `history` 占位符
   - 自动保存新的对话消息到历史记录

2. **提示词模板结构**:
   ```python
   ChatPromptTemplate.from_messages([
       ("system", "参考资料{context}"),           # 检索到的文档
       ("system", "对话历史记录如下："),
       MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
       ("user", "请回答：{input}")               # 用户当前问题
   ])
   ```

3. **数据流转过程**:
   ```
   输入: {"input": "我身高180厘米，尺码推荐"}
        ↓
   format_for_retriever: 提取 "我身高180厘米，尺码推荐"
        ↓
   retriever.invoke(): 检索相关文档 → [Document, ...]
        ↓
   format_docs(): 格式化 → "文档片段：...\n元数据：...\n"
        ↓
   format_for_prompt_template: 
   {
       "input": "我身高180厘米，尺码推荐",
       "context": "文档片段：...",
       "history": [历史消息列表]  # 由 RunnableWithMessageHistory 自动注入
   }
        ↓
   ChatPromptTemplate: 构建完整提示词
        ↓
   ChatTongyi: 生成回答
   ```

### 6. 问答界面 (`app_qa.py`)

**核心功能**: Streamlit Web 问答界面

**历史记录调用流程**:

```python
# 1. 初始化 RAG 服务（包含历史记录管理）
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService()

# 2. 调用链时传入 session_config
result = st.session_state["rag_service"].chain.stream(
    {"input": prompt}, 
    config=config.session_config  # 包含 session_id
)
```

**历史记录工作原理**:

1. **Session ID 管理**:
   - `config.session_config = {"configurable": {"session_id": "user_001"}}`
   - 所有对话使用相同的 session_id，实现历史记录共享

2. **自动历史注入**:
   - `RunnableWithMessageHistory` 根据 `session_id` 调用 `get_history("user_001")`
   - `get_history` 返回 `FileChatMessageHistory` 实例
   - 自动读取 `chat_history/user_001` 文件中的历史消息
   - 将历史消息注入到 prompt 的 `history` 占位符

3. **历史记录更新**:
   - 每次对话后，`RunnableWithMessageHistory` 自动调用 `add_messages()`
   - 将用户输入和 AI 回复保存到历史文件
   - 下次对话时自动加载

**流式输出实现**:
```python
def capture(generator, cache_list):
    for chunk in generator:
        cache_list.append(chunk)  # 缓存完整内容
        yield chunk  # 流式输出

st.chat_message("assistant").write_stream(
    capture(result, ai_res_list)
)
# 流式显示的同时，缓存完整内容用于保存历史
```

### 7. 文件上传界面 (`app_file_uploader.py`)

**核心功能**: 通过 Web 界面上传知识库文件

**工作流程**:
```
用户上传文件 → 读取文件内容 → MD5 校验 → 文本切分 → 向量化 → 存入 Chroma
```

## 🔄 完整 RAG 流程

```
┌─────────────────┐
│   用户提问      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RunnableWith   │
│  MessageHistory │ ← 自动注入历史记录
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   向量检索      │ ← 从 Chroma 检索相关文档
│  (Retriever)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   格式化文档    │
│  (format_docs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   构建提示词    │ ← 包含：context + history + input
│  (Prompt)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   调用大模型    │
│  (ChatTongyi)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   返回答案      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   保存历史      │ ← 自动保存到文件
└─────────────────┘
```

## 🚀 使用说明

### 环境配置

1. **安装依赖**:
   ```bash
   pip install langchain langchain-community langchain-chroma
   pip install streamlit dashscope
   ```

2. **配置 API Key**:
   - 设置环境变量 `DASHSCOPE_API_KEY`（阿里云百炼 API Key）

### 启动应用

1. **启动问答界面**:
   ```bash
   streamlit run app_qa.py
   ```

2. **启动文件上传界面**:
   ```bash
   streamlit run app_file_uploader.py
   ```

### 使用流程

1. **上传知识库**:
   - 访问文件上传界面
   - 上传 `.txt` 格式的知识库文件
   - 系统自动进行向量化和存储

2. **进行问答**:
   - 访问问答界面
   - 输入问题，系统自动检索相关知识并生成回答
   - 对话历史自动保存，支持上下文理解

## 🔧 配置修改

所有配置集中在 `config_data.py` 文件中：

- **修改向量库名称**: 修改 `chroma_collection_name`
- **修改检索数量**: 修改 `similarity_threshold`（k值）
- **修改文本切分大小**: 修改 `chunk_size` 和 `chunk_overlap`
- **修改模型**: 修改 `embedding_model` 和 `chat_model`
- **修改会话ID**: 修改 `session_config` 中的 `session_id`

## 📝 技术要点总结

1. **历史记录管理**: 通过 `RunnableWithMessageHistory` + `FileChatMessageHistory` 实现持久化对话历史
2. **向量检索**: 使用 Chroma + DashScope Embeddings 实现语义检索
3. **去重机制**: MD5 哈希值防止重复上传相同内容
4. **文本切分**: RecursiveCharacterTextSplitter 智能切分，保持上下文连贯
5. **流式输出**: Streamlit 的 `write_stream` 实现实时响应
6. **配置集中管理**: 所有配置统一在 `config_data.py` 中管理

## 🎯 项目特色

- ✅ 完整的 RAG 实现（检索 + 生成）
- ✅ 持久化对话历史（基于文件存储）
- ✅ 智能去重机制（MD5 校验）
- ✅ Web 界面（Streamlit）
- ✅ 流式输出（实时响应）
- ✅ 配置集中管理

---

**开发者**: Beamus Wayne  
**技术栈**: LangChain + Chroma + DashScope + Streamlit
