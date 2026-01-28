# RAG 学习与实践项目

本项目是一个完整的 RAG（Retrieval-Augmented Generation，检索增强生成）学习和实践项目，涵盖了从基础概念到实际应用的完整技术栈。

## 📚 项目结构

```
RAG/
├── LangChain_Tutorial_Fast/    # LangChain 快速教程
├── LangChain_RAG_Proj/         # 完整 RAG 项目（生产级）
├── LangGraph/                  # LangGraph 图式编程
├── PydanticAI/                 # PydanticAI 框架实践（Agent、工具、流式、RAG 等）
├── PydanticGraph/              # Pydantic Graph 图式工作流（售货机、邮件反馈、问答图）
├── Experiment/                 # 实验性示例
├── Archive/                    # 历史代码归档
└── Data/                       # 测试数据
```

## 🎯 核心项目

### 1. LangChain_RAG_Proj（推荐）

**完整的企业级 RAG 应用**，包含：

- ✅ 知识库管理（文件上传、向量化、去重）
- ✅ 智能问答系统（RAG + 对话历史）
- ✅ Web 界面（Streamlit）
- ✅ 持久化存储（向量库 + 对话历史）

**快速开始**:
```bash
cd LangChain_RAG_Proj
streamlit run app_qa.py        # 启动问答界面
streamlit run app_file_uploader.py  # 启动文件上传界面
```

**详细文档**: 查看 [LangChain_RAG_Proj/README.md](LangChain_RAG_Proj/README.md)

### 2. LangChain_Tutorial_Fast

**LangChain 快速入门教程**，包含 31 个示例文件：

- **01-10**: 基础 LLM 和 Embedding 使用
- **11-16**: Prompt 模板和 Chat 模型
- **17-20**: Chain 和输出解析器
- **21-23**: 对话历史管理
- **24-31**: RAG 完整流程

**学习路径**:
1. 从 `01_LLM.py` 开始，了解基础 LLM 调用
2. 学习 Prompt 模板（11-16）
3. 掌握 Chain 构建（17-20）
4. 理解对话历史（21-23）
5. 实践 RAG 应用（24-31）

### 3. LangGraph

**LangGraph 图式编程示例**：
- `01_HelloLangGraph.py`: 基础图式编程
- `02_FunctionalAPI.py`: 函数式 API

### 4. PydanticAI

**PydanticAI 框架实践**（阿里云百炼 / Ollama 后端）：
- **00_Preparation**：入门与模型接入（HelloPydanticAI、Ollama）
- **01_WeatherAgent**：多工具调用（天气、Gradio 界面）
- **02_StructedOutput**：结构化输出与流式（stream_text / stream_output）
- **03_ChatApp**：FastAPI + MySQL 对话应用
- **04_BankSupport**：银行支持多轮对话
- **05_SqlGen**：SQL 生成、数据分析师、RAG（pgvector）
- **07_Workflow**：复杂工作流（机票预订多 Agent）

**快速开始**：见 [PydanticAI/README.md](PydanticAI/README.md)，从 `00_Preparation/01_HelloPydanticAI.py` 或 `02_StructedOutput/01_streamOutput.py` 跑通。

### 5. PydanticGraph

**Pydantic Graph 图式工作流**（状态机、多节点图）：
- `vending_machine.py`：售货机流程（投币 → 选品 → 购买），`python vending_machine.py`
- `vending_machine_diagram.py`：输出 Mermaid 图代码，复制到 [mermaid.live](https://mermaid.live) 查看
- `genai_email_feedback.py`：写邮件 → 用户反馈 → 重写（流式生成 + Rich 展示）
- `ai_q_and_a_graph.py`：问答图（出题 → 用户作答 → 评判 → 循环）
- `graph_example.py` / `graph_example_diagram.py`：整除图示例与 Mermaid 输出

**依赖**：`pydantic-graph`、`pydantic-ai`、`rich`，模型需配置 `DASHSCOPE_API_KEY`（genai 示例）。

## 🛠️ 技术栈

### 框架
- **LangChain**: 大语言模型应用开发框架
- **LangGraph**: 图式工作流构建
- **PydanticAI**: 类型安全的 AI 应用框架（Agent、工具、流式、RAG）
- **Pydantic Graph**: 图式工作流（状态机、多节点 DAG，与 PydanticAI 可组合）

### LLM 提供商
- **OpenAI GPT**: 商业 API
- **Ollama**: 本地大模型（qwen3, qwen3-embedding）
- **阿里云百炼**: 通义千问（qwen3-max, text-embedding-v4）

### 向量数据库
- **Chroma**: 开源向量数据库
- **InMemoryVectorStore**: 内存向量存储

### Web 框架
- **Streamlit**: 快速构建 Web 应用

## 📦 安装依赖

```bash
# 核心依赖
pip install langchain langchain-community langchain-chroma
pip install langchain-text-splitters langgraph
pip install streamlit pydantic-ai pydantic-graph

# 模型提供商
pip install dashscope  # 阿里云百炼
pip install openai     # OpenAI
pip install langchain-ollama  # Ollama

# 文档处理
pip install pypdf python-docx
```

## 🔑 环境配置

### 阿里云百炼（推荐）
```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Ollama（本地）
```bash
# 安装 Ollama: https://ollama.ai
# 拉取模型
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b
```

## 📖 学习路径

### 初学者路径
1. **基础概念** (`LangChain_Tutorial_Fast/01-10`)
   - LLM 调用
   - Embedding 生成
   - 基础 Prompt

2. **进阶应用** (`LangChain_Tutorial_Fast/11-23`)
   - Prompt 模板
   - Chain 构建
   - 对话历史

3. **RAG 实践** (`LangChain_Tutorial_Fast/24-31`)
   - 文档加载
   - 向量检索
   - 完整 RAG 流程

4. **项目实战** (`LangChain_RAG_Proj`)
   - 企业级应用
   - Web 界面
   - 生产部署

### 进阶路径
- **LangGraph**: 复杂工作流设计
- **PydanticAI**: 类型安全开发、流式输出、RAG（见 [PydanticAI/README.md](PydanticAI/README.md)）
- **PydanticGraph**: 图式状态机与多节点工作流（售货机、邮件反馈、问答图）
- **自定义组件**: 扩展 LangChain 功能

## 🚀 快速开始

### 1. 运行教程示例
```bash
cd LangChain_Tutorial_Fast
python 01_LLM.py
```

### 2. 启动 RAG 项目
```bash
cd LangChain_RAG_Proj
streamlit run app_qa.py
```

### 3. 配置知识库
1. 访问文件上传界面：`streamlit run app_file_uploader.py`
2. 上传 `.txt` 格式的知识库文件
3. 系统自动进行向量化和存储

### 4. PydanticAI / PydanticGraph
```bash
# PydanticAI 入门
cd PydanticAI/00_Preparation && python 01_HelloPydanticAI.py

# PydanticGraph 售货机
cd PydanticGraph && python vending_machine.py
```
详见 [PydanticAI/README.md](PydanticAI/README.md)。

## 📝 项目特点

- ✅ **完整教程**: 从基础到进阶的完整学习路径
- ✅ **生产级项目**: 可直接部署的企业级 RAG 应用
- ✅ **多框架支持**: LangChain、LangGraph、PydanticAI、Pydantic Graph
- ✅ **多模型支持**: OpenAI、Ollama、阿里云百炼
- ✅ **Web 界面**: Streamlit 快速原型开发
- ✅ **图式工作流**: LangGraph / Pydantic Graph 状态机与 DAG 示例
- ✅ **详细文档**: 各子项目配有 README 与学习路线

## 📚 文档资源

- **LangChain_RAG_Proj**: [详细技术文档](LangChain_RAG_Proj/README.md)
- **PydanticAI**: [示例与学习路线](PydanticAI/README.md)
- **LangChain 官方文档**: https://python.langchain.com
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph
- **PydanticAI 文档**: https://ai.pydantic.dev
- **Pydantic Graph 文档**: https://graph.pydantic.dev

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅用于学习和研究目的。

---

**开发者**: Beamus Wayne  
**最后更新**: 2026-01-28
