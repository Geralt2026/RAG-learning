# Logfire 配置指南：尽可能详尽地记录

[Logfire](https://logfire.pydantic.dev/) 基于 OpenTelemetry，用于观测 PydanticAI、FastAPI、HTTP、数据库等。下面按「尽量多记、少丢」的方式配置。

---

## 一、前置：开通与鉴权

1. **安装**  
   `pip install logfire`

2. **账号与项目**  
   - 打开 [logfire.pydantic.dev](https://logfire.pydantic.dev/) 注册/登录  
   - 新建或进入一个 Project  

3. **鉴权（二选一即可）**  
   - **开发**：在项目根目录执行  
     ```bash  
     logfire auth  
     logfire projects use <你的项目名>  
     ```  
     会写 `~/.logfire/default.toml` 或当前目录 `.logfire/`。  
   - **生产/脚本**：在 Logfire 里生成 Write Token，然后：  
     ```bash  
     export LOGFIRE_TOKEN="<你的 Write Token>"  
     ```  
     或代码里 `logfire.configure(token="...")`。

---

## 二、`logfire.configure()`：发送与级别

要让数据真的发到 Logfire、并且尽量多记，建议：

```python
import logfire

# 方案 A：有 token 才发送（适合开发时按需开/关）
logfire.configure(
    send_to_logfire="if-token-present",  # 无 token 时只打本地 console，不报错
    min_level="trace",                   # 最低级别，trace/debug/info 都会记
    inspect_arguments=True,              # 默认在 3.11+ 为 True，便于 f-string 等
)

# 方案 B：强制发送到 Logfire（需已设置 LOGFIRE_TOKEN 或下面传 token）
logfire.configure(
    send_to_logfire=True,
    token=os.getenv("LOGFIRE_TOKEN"),   # 可不写，用环境变量
    service_name="my-pydantic-ai",      # 在 Logfire 里区分服务
    service_version="1.0.0",
    environment=os.getenv("LOGFIRE_ENVIRONMENT", "dev"),
    min_level="trace",
    # 控制台也看全一点
    console=logfire.ConsoleOptions(
        verbose=True,
        min_log_level="trace",
        include_timestamps=True,
        include_tags=True,
        span_style="show-parents",
    ),
)
```

**和「详尽」相关的参数简述：**

| 参数 | 作用 | 建议（尽量详尽） |
|------|------|------------------|
| `send_to_logfire` | 是否发到 Logfire | `True` 或 `"if-token-present"` |
| `min_level` | 低于此级别的 log/span 不记 | `"trace"` 或 `"debug"` |
| `inspect_arguments` | 是否做参数检测（f-string 等） | `True`（3.11+ 默认 True） |
| `console` | 本地终端输出 | `ConsoleOptions(verbose=True, min_log_level="trace")` 便于本地排查 |
| `scrubbing` | 脱敏 | 默认即可；关掉会更多明文，仅调试用 |

---

## 三、PydanticAI：`instrument_pydantic_ai()`

要「尽可能详尽」记 Agent 的输入输出与工具调用：

```python
logfire.configure(send_to_logfire=True, min_level="trace")  # 或 "if-token-present"

# 记录：prompt、completion、工具入参/返回值、二进制内容（如图片）
logfire.instrument_pydantic_ai(
    include_content=True,          # 默认 True：记 prompt、completion、tool 内容
    include_binary_content=True,   # 默认 True：记 base64 等（需 PydanticAI 0.2.5+）
    version=3,                     # 推荐用 3（需 PydanticAI 0.7.5+）
)
```

不传 `obj` 时会对所有 Agent 生效；若只对某个 agent 生效，可传：  
`logfire.instrument_pydantic_ai(agent)`。

---

## 四、FastAPI：`instrument_fastapi()`

若要连请求头、参数、端点都记全：

```python
import fastapi
import logfire

app = fastapi.FastAPI()

logfire.configure(send_to_logfire=True, min_level="trace")
logfire.instrument_pydantic_ai()

logfire.instrument_fastapi(
    app,
    capture_headers=True,      # 请求/响应头（含 cookie 等，注意脱敏）
    extra_spans=True,           # “FastAPI arguments”“endpoint function” 等 span
    record_send_receive=True,   # ASGI 收发 span（较多，默认 debug 级）
)
```

---

## 五、HTTP 调用：`instrument_httpx()`

PydanticAI 里用到的 `AsyncOpenAI` 底层往往是 httpx。对「发出去的请求和收到的响应」要详尽，可对同一个 client 做 instrument：

```python
import httpx
import logfire

# 若有全局/共享 client（例如给 OpenAI 用的）
client = httpx.AsyncClient(...)
logfire.instrument_httpx(
    client,
    capture_all=True,   # 等价于 headers + request_body + response_body
)
# 或逐项开启：
# logfire.instrument_httpx(client, capture_headers=True,
#                          capture_request_body=True, capture_response_body=True)
```

若不做 per-client，可  
`logfire.instrument_httpx()`  
不传 client，对所有 httpx 请求生效（注意和其它 OTEL 的重复封装）。

你仓库里已有类似写法（如 `WeatherAgent.py`）：

```python
logfire.instrument_httpx(http_client, capture_all=True)
```

---

## 六、数据库：MySQL / SQLite

- **MySQL（如 chat_app 用 pymysql）**：  
  在拿到 connection 之后、执行 SQL 之前：  
  ```python  
  con = pymysql.connect(...)  
  con = logfire.instrument_mysql(con)  
  ```  
  这样每条 SQL 会有一个 span，便于排查慢查询和调用链。

- **SQLite**：  
  ```python  
  import sqlite3  
  con = sqlite3.connect("...")  
  con = logfire.instrument_sqlite3(con)  
  ```

---

## 七、OpenAI 客户端（可选）

若除 PydanticAI 外还直接用 `openai` 发请求，可单独对 OpenAI 做一层观测：

```python
from openai import AsyncOpenAI
import logfire

client = AsyncOpenAI(...)
logfire.instrument_openai(client)
```

PydanticAI 已通过 `instrument_pydantic_ai` 记录模型调用，一般不必重复 `instrument_openai`，除非你有绕过 PydanticAI 的裸 OpenAI 调用。

---

## 八、Pydantic 校验（可选）

希望「每次模型校验」都有 trace（成功/失败都记）：

```python
logfire.instrument_pydantic(record="all")   # 默认即 all
# record 可选：'all' | 'failure' | 'metrics' | 'off'
```

---

## 九、在你项目里的「尽量详尽」示例

以 **03_ChatApp** 为例，把现有：

```python
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()
# …
logfire.instrument_fastapi(app)
```

改成「尽量详尽且可关」的写法，例如：

```python
import os
import logfire

# 有 token 才发到 Logfire，否则只打 console
logfire.configure(
    send_to_logfire="if-token-present",
    min_level="debug",
    service_name="chat-app",
    environment=os.getenv("LOGFIRE_ENVIRONMENT", "dev"),
    console=logfire.ConsoleOptions(
        verbose=True,
        min_log_level="debug",
        include_timestamps=True,
        include_tags=True,
    ),
)
logfire.instrument_pydantic_ai(
    include_content=True,
    include_binary_content=True,
)
logfire.instrument_fastapi(
    app,
    capture_headers=True,
    extra_spans=True,
)
# 若 chat_app 里有用到 httpx client，再对该 client 调 instrument_httpx(..., capture_all=True)
# 若用 MySQL，在 _connect() 里对 con 调用 logfire.instrument_mysql(con)
```

**01_WeatherAgent** 已对 httpx 使用 `capture_all=True`，只需把  
`logfire.configure(send_to_logfire=False)`  
改为 `send_to_logfire="if-token-present"` 并设好 `min_level`（如 `"debug"`），即可在不配 token 时本地跑、配好 token 后详尽上传。

---

## 十、环境变量速查

| 变量 | 含义 |
|------|------|
| `LOGFIRE_TOKEN` | Write Token，用于向 Logfire 发数据 |
| `LOGFIRE_SEND_TO_LOGFIRE` | `true` / `false`，是否发送 |
| `LOGFIRE_SERVICE_NAME` | 服务名 |
| `LOGFIRE_ENVIRONMENT` | 环境（如 dev / staging / prod） |
| `LOGFIRE_MIN_LEVEL` | 最低级别，如 `trace` / `debug` |
| `LOGFIRE_CONSOLE` | `false` 可关掉本地 console 输出 |
| `LOGFIRE_HTTPX_CAPTURE_ALL` | `true` 时，httpx 默认 capture_all 视为 True |

---

## 十一、简要对照：当前 vs 尽量详尽

| 项目 | 你目前常见写法 | 尽量详尽时 |
|------|----------------|------------|
| 发送 | `send_to_logfire=False` | `True` 或 `"if-token-present"` |
| 级别 | 未设（多为 info） | `min_level="trace"` 或 `"debug"` |
| PydanticAI | `instrument_pydantic_ai()` | 同上，并显式 `include_content=True`、`include_binary_content=True`、`version=3` |
| FastAPI | `instrument_fastapi(app)` | 加上 `capture_headers=True`、`extra_spans=True` |
| HTTP | 有处用 `instrument_httpx(..., capture_all=True)` | 对所有相关 client 保持 `capture_all=True` |
| 数据库 | 未 instrument | MySQL: `instrument_mysql(con)`；SQLite: `instrument_sqlite3(con)` |
| 控制台 | 默认 | `console=ConsoleOptions(verbose=True, min_log_level="trace")` |

按上面逐项打开后，再在 Logfire 后台按服务名、环境、时间范围筛选，即可做「尽可能详尽的记录」排查与复盘。
