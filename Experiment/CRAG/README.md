# CRAG：Corrective Retrieval Augmented Generation

本文档基于论文 **《Corrective Retrieval Augmented Generation》(arXiv:2401.15884v3)**，说明其提出的改良 RAG 方法，并给出在本项目中实现“基于知识库的问答机器人”的算法步骤与设计说明。

---

## 一、论文与 CRAG 简介

### 1.1 背景与动机

- **大模型幻觉**：LLM 仅依赖参数知识难以保证生成内容的准确性，容易产生事实性错误与幻觉。
- **传统 RAG 的局限**：检索增强生成（RAG）通过从外部知识库检索文档来增强输入，但其效果**强依赖检索结果的相关性**。若检索返回不相关或错误文档，反而会向生成器注入噪声，误导回答甚至加剧幻觉。
- **现有做法的不足**：多数 RAG 会**无差别地使用**检索到的文档，且常把**整篇文档**当作参考，而文档中往往包含大量与问题无关的冗余内容。

CRAG 针对“**检索出错时如何自我纠正**”这一问题，提出一套可插拔的纠正策略，在保持与各类 RAG 兼容的前提下，提升生成的鲁棒性与知识利用效率。

### 1.2 CRAG 的核心思想

1. **先评估、再决策**：用轻量级**检索评估器**对“问题–文档”相关性打分，得到整体置信度，再据此触发不同**动作**（Correct / Incorrect / Ambiguous）。
2. **三种动作**：
   - **Correct**：认为检索结果可靠 → 对文档做**知识精炼**（分解–过滤–重组），只保留关键信息。
   - **Incorrect**：认为检索结果不可靠 → **弃用**检索结果，改用**网络搜索**等外部知识进行纠正。
   - **Ambiguous**：置信度介于两者之间 → 同时使用**精炼后的检索知识 + 网络搜索知识**，软性融合。
3. **知识精炼**：对单篇文档做 **Decompose-then-Recompose**（分解–筛选–重组），按“条”（strip）评估相关性，过滤无关条、保留相关条并重组，形成更干净的内部知识。
4. **网络搜索扩展**：当检索被判为 Incorrect（或 Ambiguous 时作为补充）时，将问题改写成搜索查询，调用网络搜索（优先 Wikipedia 等权威来源），并对爬取内容同样做知识精炼，得到外部知识。

整体上，CRAG 是**即插即用**的：不改变底层检索器与生成器的接口，只在“检索结果 → 生成器输入”之间插入“评估 + 动作 + 知识精炼/网络搜索”的管道。

---

## 二、CRAG 相对传统 RAG 的改进

| 维度 | 传统 RAG | CRAG |
|------|----------|------|
| 检索结果使用 | 通常无差别全部喂给生成器 | 先评估相关性，再决定用/不用/怎么用 |
| 文档粒度 | 多以整篇文档为单位 | 分解为“知识条”，按条过滤后重组 |
| 检索失败时 | 仍用错误文档，易导致幻觉 | Incorrect 时弃用检索，改用网络搜索 |
| 置信不确定时 | 无专门策略 | Ambiguous：内部精炼知识 + 外部搜索知识结合 |

因此，CRAG 的改进可以概括为：**检索质量评估 + 基于置信度的动作分支 + 知识精炼（Decompose-then-Recompose）+ 网络搜索兜底与补充**。

---

## 三、算法流程（论文 Algorithm 1 对应）

**输入**：用户问题 \(x\)，检索得到的文档集合 \(D = \{d_1, d_2, \ldots, d_k\}\)。  
**输出**：生成回答 \(y\)。

**涉及组件**：  
- \(E\)：检索评估器  
- \(W\)：查询改写器（用于网络搜索）  
- \(G\)：生成器  

**步骤**：

1. **评估**：对每个 \((x, d_i)\) 用 \(E\) 打分，得到 \(\text{score}_i\)。
2. **置信度与动作**：根据 \(\{\text{score}_1, \ldots, \text{score}_k\}\) 得到整体判断，置信度取三者之一：`[CORRECT]` / `[INCORRECT]` / `[AMBIGUOUS]`（通过上、下阈值划分）。
3. **知识准备**：
   - 若 **Correct**：\(k \leftarrow \text{Knowledge\_Refine}(x, D)\)（仅用精炼后的内部知识）。
   - 若 **Incorrect**：\(k \leftarrow \text{Web\_Search}(W(x))\)（仅用外部搜索知识，\(W\) 将问题改写成搜索查询）。
   - 若 **Ambiguous**：\(k \leftarrow \text{Knowledge\_Refine}(x, D) + \text{Web\_Search}(W(x))\)（内部 + 外部）。
4. **生成**：\(y \leftarrow G(x, k)\)。

其中：

- **Knowledge_Refine**：对 \(D\) 中文档做分解（切成多条）、用 \(E\) 对每条打分、过滤低分条、按顺序拼接成“内部知识”。
- **Web_Search**：用 \(W(x)\) 得到搜索查询 → 调用搜索 API → 获取页面/摘要 → 同样做分解与评估筛选，得到“外部知识”。

---

## 四、各模块说明与实现要点

### 4.1 检索评估器（Retrieval Evaluator）

- **作用**：判断“问题–文档”或“问题–知识条”的相关性，输出连续分数（如 [-1, 1]）。
- **论文实现**：基于 T5-large 微调，轻量（约 0.77B），对每个 (question, document) 单独打分。
- **实现要点**：
  - 可用小模型（如 T5、BERT 类）或轻量 LLM，输入为 `"[CLS] question [SEP] document"` 或类似格式，输出为标量分数或二分类（相关/不相关）再映射为分数。
  - 若无现成评估器，可用 LLM（如 Ollama/OpenAI）做“是否包含回答问题所需信息”的判别，并将 yes/no 映射为分数（例如 1 / -1），用于阈值判断。

### 4.2 动作触发（Action Trigger）

- **输入**：各文档的 score（或各知识条的 score 的聚合，如 max/mean）。
- **逻辑**：
  - 若 **任一** score > 上阈值 → **Correct**。
  - 若 **全部** score < 下阈值 → **Incorrect**。
  - 否则 → **Ambiguous**。
- **阈值**：论文中按数据集设不同值（如 PopQA 用 0.59 / -0.99）；本项目中可先设一组默认值，再按验证集调参。

### 4.3 知识精炼（Knowledge Refinement / Decompose-then-Recompose）

- **分解**：将每篇文档切成“条”（strip）。短文可 1–2 句一条；长文按句或按固定 token 数切分为若干段，每段视为一条。
- **打分与过滤**：对每条 strip，用同一评估器 \(E\) 计算与问题 \(x\) 的 score；去掉 score 低于某阈值的条（论文中如 -0.5），或保留 top-k 条（如 top-5）。
- **重组**：将保留的条按原顺序拼接，得到“内部知识”文本，作为后续生成上下文。

### 4.4 网络搜索（Web Search）

- **查询改写**：用 \(W\) 将用户问题 \(x\) 改写成适合搜索引擎的短查询（论文用 LLM 抽 2–3 个关键词）。
- **搜索与抓取**：调用搜索 API（如 Serper、Google Custom Search、Bing 等）得到 URL 列表；优先选 Wikipedia 等权威站点，抓取正文或摘要。
- **精炼**：对抓取到的文本做与 4.3 相同的“分解–打分–过滤–重组”，得到“外部知识”。

若本地仅做知识库问答、不做真实网络搜索，可将“Web Search”替换为：从另一更大知识库/维基摘要库中再检索一轮，或返回“当前知识库未覆盖，建议联网搜索”的占位逻辑，以保留 CRAG 的决策结构。

### 4.5 生成器（Generator）

- 输入：用户问题 \(x\) + 上述得到的知识 \(k\)（可能为内部知识、外部知识或两者拼接）。
- 使用任意兼容的 RAG 生成接口即可（如 LangChain / LangGraph / PydanticAI / Agno 的 RAG 链），只需保证 prompt 中包含 \(k\) 作为 context。

---

## 五、本项目实现步骤与设计

### 5.1 目标

实现一个**基于知识库的问答机器人**（命令行即可），采用 CRAG 的“评估 → 动作 → 知识准备 → 生成”流程，可选地接入网络搜索或“二次检索”作为 Incorrect/Ambiguous 的补充。

### 5.2 技术选型建议

- **框架**：LangChain / LangGraph / PydanticAI / Agno 任选，用于：检索（向量库）、链/图编排、与 LLM 对话。
- **知识库**：本地向量库（如 Chroma、FAISS）+ 文本切分（RecursiveCharacterTextSplitter 等），与现有 `LangChain_RAG_Proj` 类似。
- **检索评估器**：
  - **简化版**：用同一 LLM（Ollama/OpenAI）对“(问题, 文档)”做 yes/no 判断并映射为分数，无需单独训练模型。
  - **完整版**：使用在 (question, document) 数据上微调的小模型（如 sentence-transformers + 分类头）输出相关性分数。
- **生成器**：与现有 RAG 项目一致，使用 Ollama 或 OpenAI 等，以“问题 + 精炼后的知识”为 context 生成回答。

### 5.3 实现步骤（推荐顺序）

1. **基础 RAG 管道**  
   - 加载本地知识库（文档切分、向量化、建索引）。  
   - 实现：用户问题 → 检索 top-k 文档 → 拼成 context → 调用 LLM 生成。  
   - 确保单轮问答在命令行可跑通。

2. **检索评估器**  
   - 实现接口：`score = evaluator(question, document)`，返回 float。  
   - 简化实现：用 LLM 对 “问题 + 文档片段” 做“是否包含回答问题所需信息”的判断，映射为 1.0 / -1.0（或 0/1 再缩放）。  
   - 可选：对“文档条”也复用同一接口，便于后续知识精炼。

3. **动作触发**  
   - 输入：每个检索文档的 score。  
   - 实现：根据上、下阈值得到 Correct / Incorrect / Ambiguous；  
   - 输出：动作类型 + 用于下一步的“待精炼文档集合”和“是否需要网络搜索”的标志。

4. **知识精炼（Decompose-then-Recompose）**  
   - 对“待精炼文档”按句或按块切分为 strips；  
   - 对每条 strip 调用评估器打分；  
   - 过滤低分条（或取 top-k），按顺序拼接成一段“精炼知识”字符串。  
   - 该字符串作为“内部知识”传入生成器。

5. **网络搜索（或替代方案）**  
   - **若有 API**：实现查询改写 \(W(x)\)（可用同一 LLM）+ 搜索 API + 结果抓取/摘要；对结果做与步骤 4 相同的精炼，得到“外部知识”。  
   - **若无 API**：用“二次检索”（从更大知识库或不同 chunk 再检一轮）或固定提示“知识库未覆盖该问题”作为 Incorrect/Ambiguous 的替代。

6. **CRAG 主流程编排**  
   - 串联：检索 → 评估 → 动作判断 →  
     - Correct：仅知识精炼 → 生成；  
     - Incorrect：仅外部知识（或替代）→ 生成；  
     - Ambiguous：精炼知识 + 外部知识拼接 → 生成。  
   - 在命令行读入用户问题，输出最终回答；可同时打印当前动作（Correct/Incorrect/Ambiguous）便于调试。

7. **调参与扩展**  
   - 调节上/下阈值、精炼时的 strip 长度与 top-k、检索的 top-k；  
   - 可选：记录每轮动作与分数，用于分析检索质量与 CRAG 行为。

### 5.4 模块与数据流（设计草图）

```
用户问题 x
    ↓
[检索器] → 文档集合 D
    ↓
[评估器 E] → 每个 (x, d_i) 的 score_i
    ↓
[动作触发] → Correct / Incorrect / Ambiguous
    ↓
  ┌─────────┼─────────┐
  ↓         ↓         ↓
Correct  Incorrect  Ambiguous
  ↓         ↓         ↓
Refine(D)  Web(W(x))  Refine(D) + Web(W(x))
  ↓         ↓         ↓
  └─────────┴─────────┘
            ↓
        知识 k
            ↓
    [生成器 G] → 回答 y
```

### 5.5 文件与目录建议（CRAG 文件夹下）

- `README.md`：本文档（论文解读 + 算法与实现说明）。  
- `config.py` 或 `config.yaml`：阈值、top-k、评估器/生成器模型名、是否启用网络搜索等。  
- `evaluator.py`：检索评估器封装（LLM 或小模型）。  
- `refinement.py`：知识精炼（分解、打分、过滤、重组）。  
- `search.py`（可选）：查询改写 + 网络搜索/二次检索 + 结果精炼。  
- `crag.py` 或 `pipeline.py`：CRAG 主流程（检索 → 评估 → 动作 → 知识准备 → 生成）。  
- `cli.py` 或 `main.py`：命令行入口，读入问题、调用 pipeline、打印回答与（可选）动作类型。  
- `requirements.txt`：依赖（如 langchain、chromadb、ollama/openai 等）。

---

## 六、参考文献与资源

- 论文：**Corrective Retrieval Augmented Generation** (Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling), arXiv:2401.15884v3.  
- 官方代码（论文中提及）：[github.com/HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)。

---

## 七、本项目实现概览与文件结构

本目录下已实现完整 CRAG 流程，支持 **FastAPI** 与 **Streamlit** 两种交互方式；知识库路径为 **`Experiment/CRAG/Files`**，用户将 PDF 放入该目录后通过「构建知识库」即可使用。PDF 的文本识别与版面分析使用 **MinerU** 完成。

| 文件/目录 | 说明 |
|-----------|------|
| `config.py` | 路径、阈值、LLM/Embedding 提供商与模型名等配置 |
| `pdf_loader.py` | 基于 MinerU：PDF 每页转图像 → 版面抽取 → 文本块整理为 Document |
| `vector_store.py` | Chroma 向量库封装，Ollama/OpenAI 等 Embedding |
| `knowledge_base.py` | 从 `Files/` 加载 PDF（MinerU）→ 切分 → 写入向量库 |
| `evaluator.py` | 检索评估器：PydanticAI 模型判断「问题-文档」是否相关，输出 1.0 / -1.0 |
| `refinement.py` | 知识精炼：分解 → 打分 → 过滤 → 重组（Decompose-then-Recompose） |
| `action_trigger.py` | 根据分数触发 Correct / Incorrect / Ambiguous |
| `search.py` | 外部知识：查询改写 + 网络搜索占位（可扩展真实 API） |
| `agent_model.py` | **PydanticAI** 模型与 Agent（qwen3-max/通义），供评估与生成 |
| `crag.py` | CRAG 主流程：检索 → 评估 → 动作 → 知识准备 → 生成（LLM 走 PydanticAI） |
| `api.py` | FastAPI：`/ask` 问答、`/build_kb` 重建知识库 |
| `app_streamlit.py` | Streamlit 交互界面：输入问题、查看回答与动作类型、侧栏重建知识库 |
| `Files/` | **知识库 PDF 目录**：用户在此放入 PDF，构建知识库时自动识别 |
| `chroma_db/` | 向量库持久化目录（自动生成） |
| `requirements.txt` | Python 依赖列表 |

---

## 八、安装与运行

### 8.1 环境要求

- Python ≥ 3.10
- 若使用 MinerU 做 PDF 识别：需 GPU 且已安装 PyTorch、transformers ≥ 4.56.0
- 若仅用 Ollama 本地模型：需已安装 [Ollama](https://ollama.com/) 并拉取所用模型（如 `qwen3:4b`、`nomic-embed-text`）

### 8.2 安装依赖

在项目根目录或 `Experiment/CRAG` 下执行：

```bash
cd d:\Test\RAG\Experiment\CRAG
pip install -r requirements.txt
```

若使用 Ollama 且不需要 MinerU（例如先用已有向量库或纯文本导入），可先不装 MinerU 相关依赖（`mineru-vl-utils`、`modelscope`、`transformers` 等），仅安装 LangChain、Chroma、FastAPI、Streamlit 等；此时「构建知识库」需通过其他方式导入文档（或先跳过，仅测试 API/Streamlit 与 CRAG 流程）。

使用 MinerU 时建议：

```bash
pip install torch transformers modelscope mineru-vl-utils PyMuPDF
```

### 8.3 配置

- 编辑 **`config.py`** 或通过环境变量：
  - `CRAG_LLM_PROVIDER`：`ollama` / `openai` / `dashscope`
  - `CRAG_EMBEDDING_PROVIDER`：`ollama` / `openai` / `dashscope`
  - `CRAG_OLLAMA_LLM`、`CRAG_OLLAMA_EMBED`：Ollama 模型名
  - `OPENAI_API_KEY`、`OPENAI_API_BASE`（若用 OpenAI 或兼容 API）
- 知识库路径固定为 **`Experiment/CRAG/Files`**，将 PDF 放入该目录即可。
- **PDF 识别强制使用 MinerU**（无 PyMuPDF 回退）。FastAPI 的 `/build_kb` 会在**子进程**中执行构建，避免在主进程加载 MinerU 时出现依赖冲突（如 `No module named 'frontend'`）。

### 8.4 构建知识库

1. 将需要入库的 **PDF** 放入 **`Experiment/CRAG/Files`**。
2. 通过以下任一方式触发「构建知识库」：
   - **Streamlit**：侧边栏点击「重新构建知识库」。
   - **FastAPI**：请求 `POST /build_kb?force_rebuild=true`。
  也可在 Streamlit 侧栏或 FastAPI `/build_kb` 中操作。

构建过程会使用 MinerU 对每个 PDF 逐页做版面分析与文本抽取，切分后写入 Chroma。

### 8.5 启动交互接口

**FastAPI（推荐用于接口对接 / 前端调用）：**

```bash
cd d:\Test\RAG\Experiment\CRAG
uvicorn api:app --host 0.0.0.0 --port 8000
```

- 问答：`POST /ask`，Body `{"question": "你的问题"}`。
- 重建知识库：`POST /build_kb?force_rebuild=true`。
- 健康检查：`GET /health`。

**Streamlit（推荐用于本地试玩）：**

```bash
cd d:\Test\RAG\Experiment\CRAG
streamlit run app_streamlit.py
```

浏览器打开提示的地址（通常 http://localhost:8501），在输入框输入问题即可；可展开「查看使用的参考资料」查看 CRAG 实际使用的上下文与动作类型。

---

## 九、操作说明（简要）

1. **准备知识库**：将 PDF 放入 `Experiment/CRAG/Files`，执行一次「构建知识库」。
2. **启动服务**：按需启动 FastAPI 或 Streamlit（见 8.5）。
3. **提问**：在 Streamlit 输入框、对 FastAPI `/ask` 发送问题，或运行 `python cli.py` 在命令行提问；系统会按 CRAG 流程：检索 → 评估 → 触发 Correct/Incorrect/Ambiguous → 精炼或补充知识 → 生成回答。
4. **更新知识库**：新增或更换 PDF 后，再次执行「构建知识库」（建议使用 `force_rebuild=true` 或界面「重新构建知识库」）。

### 常见问题

- **FastAPI 调用 `/build_kb` 报 500 或提示 `No module named 'frontend'`**  
  构建知识库**强制使用 MinerU**。为避免在主进程（FastAPI/uvicorn）中加载 MinerU 时触发依赖冲突，`/build_kb` 会在**独立子进程**中执行 `build_knowledge_base`。请确保从 **`Experiment/CRAG`** 目录启动 uvicorn（例如 `cd Experiment/CRAG && uvicorn api:app --host 0.0.0.0 --port 8000`），以便子进程能正确找到模块。若仍报错，可先在终端执行：  
  `cd Experiment/CRAG && python -c "from knowledge_base import build_knowledge_base; print(build_knowledge_base(force_rebuild=True))"`  
  确认 MinerU 与 modelscope 已正确安装且无缺失依赖。

本文档为 CRAG 算法在“基于知识库的问答机器人”中的实现说明；当前实现已包含检索评估、动作触发、知识精炼与占位式外部知识，并按 MinerU 强制完成 PDF 识别。
