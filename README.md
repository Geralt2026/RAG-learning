# RAG 学习与实践项目

本项目是一个完整的 RAG（Retrieval-Augmented Generation，检索增强生成）学习和实践项目，涵盖了从基础概念到实际应用的完整技术栈，并包含多框架 Agent、图式工作流与实验性 RAG 改进（如 CRAG）。

## 📚 项目结构

```
RAG-learning/
├── Agents/                    # 多模态 Agent（Agno + Gemini / Ollama）
│   └── multimodal_agent/      # 多模态推理、视频理解等
├── Agno/                      # Agno 框架入门与助手
│   ├── 00_Get_Started/        # HelloAgno、First Agent、Learning、Agentic/CrossUser
│   └── 01_Assist_Agent/       # 助手型 Agent
├── LangChain_Tutorial_Fast/   # LangChain 快速教程（31 个示例）
├── LangChain_RAG_Proj/        # 完整 RAG 项目（生产级）
├── LangGraph/                 # LangGraph 图式编程与练习
├── PydanticAI/                # PydanticAI 框架实践（Agent、工具、流式、RAG、AG-UI、工作流）
├── PydanticGraph/             # Pydantic Graph 图式工作流（售货机、邮件反馈、问答图）
├── Experiment/                # 实验性示例
│   ├── CRAG/                  # 纠正式 RAG（论文实现，FastAPI + Streamlit，MinerU PDF）
│   ├── VideoCut/              # 智能视频合成与防重复（分镜脚本、素材库、FFmpeg）
│   └── 其他示例               # weather_agent、stream_whales、quantqmt 等
├── Archive/                   # 历史代码归档
└── Data/                      # 测试数据（JSON、TXT）
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

- **01-10**: 基础 LLM 和 Embedding 使用（含 Ollama、流式输出）
- **11-16**: Prompt 模板和 Chat 模型（zero-shot、few-shot、ChatPromptTemplate）
- **17-20**: Chain 和输出解析器（StrOutputParser、JsonOutputParser、RunnableLambda）
- **21-23**: 对话历史管理（InMemory、FileChatMessageHistory）
- **24-31**: RAG 完整流程（DocumentLoaders、TextSplitter、VectorStores、RunnablePassthrough）

**学习路径**:
1. 从 `01_LLM.py` 开始，了解基础 LLM 调用
2. 学习 Prompt 模板（11-16）
3. 掌握 Chain 构建（17-20）
4. 理解对话历史（21-23）
5. 实践 RAG 应用（24-31）

### 3. LangGraph

**LangGraph 图式编程示例**：

- `00_HelloWorld_Graph.py`：Hello World 图
- `01_HelloGraph_exercise.py`：图练习
- `02_Multi_Inputs.py` / `03_Multi_Inputs_exercise.py` / `04_Multi_Inputs.py`：多输入图
- `Archive/LangGraph-Course-freeCodeCamp/`：课程笔记（Graphs、Agents、Exercises 等 Jupyter 与 PDF）

### 4. PydanticAI

**PydanticAI 框架实践**（阿里云百炼 / Ollama 后端）：

- **00_Preparation**：入门与模型接入（HelloPydanticAI、Ollama）
- **01_WeatherAgent**：多工具调用（天气、Gradio 界面）
- **02_StructedOutput**：结构化输出与流式（stream_text / stream_output）
- **03_ChatApp**：FastAPI + MySQL 对话应用
- **04_BankSupport**：银行支持多轮对话
- **05_SqlGen**：SQL 生成、数据分析师、RAG（pgvector）
- **06_AG-UI**：Agent 与 AG-UI 对接（run_ag_ui、handle_ag_ui_request）
- **07_Workflow**：复杂工作流（机票预订、question_graph）

**快速开始**：见 [PydanticAI/README.md](PydanticAI/README.md)，从 `00_Preparation/01_HelloPydanticAI.py` 或 `02_StructedOutput/01_streamOutput.py` 跑通。

### 5. PydanticGraph

**Pydantic Graph 图式工作流**（状态机、多节点图）：

- `vending_machine.py`：售货机流程（投币 → 选品 → 购买），`python vending_machine.py`
- `vending_machine_diagram.py`：输出 Mermaid 图代码，复制到 [mermaid.live](https://mermaid.live) 查看
- `genai_email_feedback.py`：写邮件 → 用户反馈 → 重写（流式生成 + Rich 展示）
- `ai_q_and_a_graph.py`：问答图（出题 → 用户作答 → 评判 → 循环）
- `graph_example.py` / `graph_example_diagram.py`：整除图示例与 Mermaid 输出

**依赖**：`pydantic-graph`、`pydantic-ai`、`rich`，模型需配置 `DASHSCOPE_API_KEY`（genai 示例）。

### 6. Agents（多模态 Agent）

**基于 Agno 的多模态 Agent**（Streamlit 界面）：

- `multimodal_agent/`：多模态 Agent（如 Gemini + 视频输入）、多模态推理 Agent
- 支持 Google Gemini、Ollama 等，可处理视频、图像与文本

**运行**：进入 `Agents/multimodal_agent`，配置 Gemini API Key 后 `streamlit run multimodal_agent.py`（或对应入口）。

### 7. Agno

**Agno 框架入门与助手型 Agent**：

- **00_Get_Started**：`00_HelloAgno.py`、`01_First_Agno_Agent.py`、`02_Learning.py`、`03_Agentic_Learning.py`、`04_CrossUser_learning.py`（含 SQLite 持久化、Ollama）
- **01_Assist_Agent**：`agno_assist.py` 助手 Agent

**依赖**：`agno`，本地模型需 Ollama（如 `qwen3-vl:4b`）。

### 8. Experiment（实验性示例）

| 子目录/文件 | 说明 |
|-------------|------|
| **CRAG/** | **纠正式 RAG**：基于论文《Corrective Retrieval Augmented Generation》实现。检索评估 → Correct/Incorrect/Ambiguous 动作 → 知识精炼或网络搜索 → 生成。FastAPI + Streamlit，PDF 使用 MinerU 做版面分析与文本抽取。详见 [Experiment/CRAG/README.md](Experiment/CRAG/README.md) |
| **VideoCut/** | **智能视频合成与防重复**：分镜脚本（YAML）驱动、云端素材库、FFmpeg 剪辑、转场/滤镜随机化、成片帧哈希防重复。含 Agent 编排、FastAPI。详见 [Experiment/VideoCut/README.md](Experiment/VideoCut/README.md) |
| `weather_agent.py` / `weather_agent_gradio.py` | 天气 Agent 示例 |
| `stream_whales.py` | 流式输出示例（如鲸鱼表格） |
| `quantqmt.py` | 量化/策略相关示例 |

---

## 🛠️ 技术栈

### 框架
- **LangChain**：大语言模型应用开发框架
- **LangGraph**：图式工作流构建
- **PydanticAI**：类型安全的 AI 应用框架（Agent、工具、流式、RAG）
- **Pydantic Graph**：图式工作流（状态机、多节点 DAG，与 PydanticAI 可组合）
- **Agno**：Agent 框架（多模态、Learning、SQLite 等）

### LLM 提供商
- **OpenAI GPT**：商业 API
- **Ollama**：本地大模型（qwen3、qwen3-embedding、qwen3-vl 等）
- **阿里云百炼**：通义千问（qwen-max、text-embedding-v4）
- **Google Gemini**：多模态（Agents 多模态示例）

### 向量数据库
- **Chroma**：开源向量数据库
- **pgvector**：PostgreSQL 扩展（PydanticAI 05_SqlGen RAG）
- **InMemoryVectorStore**：内存向量存储

### Web / 服务
- **Streamlit**：快速构建 Web 应用
- **FastAPI**：API 服务（CRAG、VideoCut、ChatApp）
- **Gradio**：天气 Agent 等界面

### 其他
- **MinerU**：PDF 版面分析与文本抽取（Experiment/CRAG）
- **FFmpeg**：视频剪辑与合成（Experiment/VideoCut）

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

# Agno / 多模态 Agent（可选）
pip install agno

# 子项目单独依赖见各目录 requirements.txt（如 Experiment/CRAG、Experiment/VideoCut）
```

## 🔑 环境配置

### 阿里云百炼（推荐）
```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Ollama（本地）
```bash
# 安装 Ollama: https://ollama.ai
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b
# 多模态可选：ollama pull qwen3-vl:4b
```

### Google Gemini（Agents 多模态）
在 Streamlit 侧栏或环境变量中配置 Gemini API Key（见 [Google AI Studio](https://aistudio.google.com/apikey)）。

## 📖 学习路径

### 初学者路径
1. **基础概念**（`LangChain_Tutorial_Fast/01-10`）：LLM 调用、Embedding、基础 Prompt
2. **进阶应用**（`LangChain_Tutorial_Fast/11-23`）：Prompt 模板、Chain、对话历史
3. **RAG 实践**（`LangChain_Tutorial_Fast/24-31`）：文档加载、向量检索、完整 RAG 流程
4. **项目实战**（`LangChain_RAG_Proj`）：企业级应用、Web 界面、生产部署

### 进阶路径
- **LangGraph**：复杂工作流设计（多输入、条件分支）
- **PydanticAI**：类型安全开发、流式输出、RAG、工作流（见 [PydanticAI/README.md](PydanticAI/README.md)）
- **PydanticGraph**：图式状态机与多节点工作流
- **Agno / Agents**：多模态 Agent、Learning、助手型 Agent
- **Experiment/CRAG**：纠正式 RAG、检索评估与知识精炼
- **Experiment/VideoCut**：分镜驱动视频合成与防重复

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
cd PydanticAI/00_Preparation && python HelloPydanticAI.py
cd PydanticGraph && python vending_machine.py
```
详见 [PydanticAI/README.md](PydanticAI/README.md)。

### 5. CRAG（纠正式 RAG）
```bash
cd Experiment/CRAG
pip install -r requirements.txt
# 将 PDF 放入 Files/ 后构建知识库，再启动：
uvicorn api:app --host 0.0.0.0 --port 8000
# 或 streamlit run app_streamlit.py
```
详见 [Experiment/CRAG/README.md](Experiment/CRAG/README.md)。

### 6. Agno 入门
```bash
cd Agno/00_Get_Started
python 00_HelloAgno.py   # 需 Ollama 与 qwen3-vl:4b
```

## 📝 项目特点

- ✅ **完整教程**：从基础到进阶的完整学习路径
- ✅ **生产级项目**：可直接部署的企业级 RAG 应用
- ✅ **多框架支持**：LangChain、LangGraph、PydanticAI、Pydantic Graph、Agno
- ✅ **多模型支持**：OpenAI、Ollama、阿里云百炼、Google Gemini
- ✅ **Web 与 API**：Streamlit、FastAPI、Gradio
- ✅ **图式工作流**：LangGraph / Pydantic Graph 状态机与 DAG 示例
- ✅ **实验性 RAG**：CRAG 纠正式检索、知识精炼与动作分支
- ✅ **多模态与 Agent**：Agno 多模态 Agent、Learning、助手型 Agent
- ✅ **详细文档**：各子项目配有 README 与学习路线

## 📚 文档资源

- **LangChain_RAG_Proj**：[详细技术文档](LangChain_RAG_Proj/README.md)
- **PydanticAI**：[示例与学习路线](PydanticAI/README.md)
- **Experiment/CRAG**：[CRAG 论文与实现说明](Experiment/CRAG/README.md)
- **Experiment/VideoCut**：[视频合成与防重复](Experiment/VideoCut/README.md)
- **LangChain**：https://python.langchain.com
- **LangGraph**：https://langchain-ai.github.io/langgraph
- **PydanticAI**：https://ai.pydantic.dev
- **Pydantic Graph**：https://graph.pydantic.dev

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅用于学习和研究目的。

---

**开发者**: Beamus Wayne  
**最后更新**: 2026-02-07
