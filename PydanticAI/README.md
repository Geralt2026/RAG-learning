# PydanticAI 示例与学习笔记

本仓库基于 [PydanticAI](https://ai.pydantic.dev/) 官方示例与文档，使用**阿里云百炼（DashScope）** 或 **Ollama** 作为模型后端，整理为可本地运行的示例与后续学习规划。

---

## 环境与依赖

- **Python** 3.10+
- **环境变量**（按示例需要配置）：
  - `DASHSCOPE_API_KEY`：阿里云百炼 API Key（多数示例必需）
  - `WEATHER_API_KEY`、`LOCATION_API_KEY`：天气智能体可选
  - `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE`：聊天应用（MySQL）

常见依赖：`pydantic-ai`、`openai`、`httpx`、`logfire`、`rich`、`fastapi`、`uvicorn`、`pymysql`、`gradio` 等，按各示例目录内说明安装。

- **Logfire 观测**：各示例默认 `logfire.configure(send_to_logfire=False)`，仅打本地日志。若需向 [Logfire](https://logfire.pydantic.dev/) 上报并**尽可能详尽记录**（含 PydanticAI / FastAPI / HTTP / DB），见 [LOGFIRE_CONFIG.md](LOGFIRE_CONFIG.md)。

---

## 目录与示例概览

### 00_Preparation — 入门与模型接入

| 文件 | 说明 | 运行方式 |
|------|------|----------|
| `01_HelloPydanticAI.py` | 使用百炼 qwen + Pydantic 输出结构（如 `city/country`）的极简示例 | `python 01_HelloPydanticAI.py` |
| `02_PydanticAI_Ollama.py` | 使用本地 Ollama（如 qwen3:4b）+ `output_type=CityLocation` | 需先启动 Ollama，再 `python 02_PydanticAI_Ollama.py` |

### 01_WeatherAgent — 多工具调用与对话式代理

| 文件 | 说明 | 运行方式 |
|------|------|----------|
| `WeatherAgent.py` | 天气智能体：`get_lat_lng` / `get_my_location` + `get_weather`，支持多城市与「我所在地」 | `python WeatherAgent.py` |
| `weather_agent_gradio.py` | 同上逻辑的 Gradio Web 界面 | `python weather_agent_gradio.py` |

### 02_StructedOutput — 结构化输出与流式

| 文件 | 说明 | 运行方式 |
|------|------|----------|
| `01_streamOutput.py` | `stream_text()` 流式纯文本（累计） | `python 01_streamOutput.py` |
| `02_streamOutput_delta.py` | `stream_text(delta=True)` 按片段输出 | `python 02_streamOutput_delta.py` |
| `03_streamed_user_profile.py` | `output_type=UserProfile`（TypedDict）+ `stream_output()` | `python 03_streamed_user_profile.py` |
| `04_streamed_markdown.py` | 流式 Markdown + Rich Live 展示 | `python 04_streamed_markdown.py` |
| `05_streamed_markdown_Ollama.py` | 同上，后端为 Ollama | `python 05_streamed_markdown_Ollama.py` |
| `06_stream_whales.py` | `output_type=list[Whale]`，流式表格（Rich Table） | `python 06_stream_whales.py` |

### 03_ChatApp — 对话式应用（FastAPI + MySQL）

| 文件 | 说明 | 运行方式 |
|------|------|----------|
| `chat_app.py` | FastAPI 服务：`/` 主页、`/chat/` 获取/发送消息，MySQL 存历史 | 在 `03_ChatApp` 下 `python chat_app.py` |
| `chat_app.html` / `chat_app.ts` | 前端页面与逻辑（TS 在浏览器中运行），Markdown 渲染依赖 marked CDN | 浏览器打开首页后由后端提供 |

---

## 学习路线与计划（参考官方示例）

下表中「状态」表示当前仓库覆盖情况，「计划」表示后续可补充的学习方向。**弃用**表示标记为不学习，原因：边缘化、学习成效比低。

| 主题 | 官方/常见示例 | 本仓库状态 | 说明或计划 |
|------|----------------|------------|------------|
| **入门与模型** | Pydantic 模型、基础 Agent | ✅ 已完成 | `00_Preparation`、`01_HelloPydanticAI`、Ollama 示例 |
| **工具与多步推理** | 天气智能体、工具编排 | ✅ 已完成 | `01_WeatherAgent`（含 IP 定位、天气 API） |
| **结构化输出** | output_type、TypedDict、流式校验 | ✅ 已完成 | `02_StructedOutput`（UserProfile、Whale 表、Markdown） |
| **流式输出** | stream_text / stream_output、Live 展示 | ✅ 已完成 | `01/02` 流式文本，`04/05/06` 流式 Markdown/表格 |
| **对话式应用** | FastAPI 聊天、历史存储 | ✅ 已完成 | `03_ChatApp`（MySQL 存历史、流式回复） |
| **银行支持** | 多轮对话、业务流程、校验 | 📋 计划 | 可做：意图识别、表单校验、话术与工具调用 |
| **SQL 生成** | 自然语言 → SQL、安全执行 | 📋 计划 | 可做：Schema 约束、只读执行、参数化防注入 |
| **数据分析师** | 上传数据、分析指令、图表建议 | 📋 计划 | 可做：pandas/小数据集 + 工具调用 + 简单可视化 |
| **RAG** | 检索增强、文档问答 | 📋 计划 | 可做：本地/向量库检索 + 引用与引用溯源 |
| **复杂工作流** | 机票预订、多步状态机 | 📋 计划 | 可做：状态与工具组合、确认与回滚 |
| **问题图（Problem Graph）** | 多节点、条件分支、子任务 | 📋 计划 | 可做：DAG 编排、子 Agent 或工具链 |
| **业务集成** | Slack/Modal 潜在客户筛选等 | ❌ 弃用 | 边缘化、学习成效比低（消息推送、定时任务、外部 API 封装等） |
| **UI 示例** | AG-UI、对话组件 | ❌ 弃用 | 边缘化、学习成效比低（Gradio 定制、前端组件与样式规范等） |

---

## 使用与扩展建议

1. **先跑通**：从 `00_Preparation/01_HelloPydanticAI.py` 和 `02_StructedOutput/01_streamOutput.py` 入手，确认环境和 `DASHSCOPE_API_KEY` 正常。
2. **按主题深入**：工具调用看 `01_WeatherAgent`，结构化与流式看 `02_StructedOutput`，Web 对话看 `03_ChatApp`。
3. **学习计划**：上表「📋 计划」项可按优先级拆成小任务（如先做 SQL 生成、再做 RAG）；「❌ 弃用（不学习）」项不纳入学习。

若某示例运行时缺少依赖，可根据报错 `pip install <包名>`，或查看该示例文件顶部的注释与文档字符串。
