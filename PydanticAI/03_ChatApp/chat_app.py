"""一个使用 FastAPI 构建的简单聊天应用示例。

运行方式：

    pip install pymysql
    python 03_ChatApp.py

MySQL 需事先创建数据库（默认名 chat_app），并通过环境变量配置：
MYSQL_HOST、MYSQL_PORT、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。
"""

from __future__ import annotations as _annotations


import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable

import pymysql
from pymysql.connections import Connection as MySQLConnection
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

# 配置日志
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

# 创建OpenAI客户端
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))


# 创建Agent
agent = Agent(model)

# 获取当前目录
THIS_DIR = Path(__file__).parent


# 生命周期管理
@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {"db": db}


# 创建FastAPI应用
app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


# 获取主页
@app.get("/")
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / "chat_app.html"), media_type="text/html")


# 获取typescript代码
@app.get("/chat_app.ts")
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / "chat_app.ts"), media_type="text/plain")


# 获取数据库
async def get_db(request: Request) -> Database:
    return request.state.db


# 获取聊天记录
@app.get("/chat/")
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b"\n".join(json.dumps(to_chat_message(m)).encode("utf-8") for m in msgs),
        media_type="text/plain",
    )


# 聊天消息格式
class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


# 转换为聊天消息
def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


# 发送聊天消息
@app.post("/chat/")
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    # 流式输出消息
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # 流式输出用户提示，以便可以立即显示
        yield (
            json.dumps(
                {
                    "role": "user",
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "content": prompt,
                }
            ).encode("utf-8")
            + b"\n"
        )
        # 获取聊天历史，以便作为上下文传递给Agent
        messages = await database.get_messages()
        # 运行Agent，使用用户提示和聊天历史
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream_output(debounce_by=0.01):
                # text here is a `str` and the frontend wants a JSON encoded ModelResponse, so we create one
                # JSON编码的ModelResponse，所以我们创建一个ModelResponse对象
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode("utf-8") + b"\n"
        # 将新的消息（例如用户提示和Agent响应）添加到数据库
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type="text/plain")  # 返回流式响应


# 参数规范
P = ParamSpec("P")
R = TypeVar("R")


# 数据库（MySQL，通过环境变量 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE 配置）
@dataclass
class Database:
    """简单的数据库，用于在 MySQL 中存储聊天消息。

    使用 PyMySQL 同步驱动，在线程池执行器中异步执行查询。
    """

    con: MySQLConnection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    # 类方法，用于连接到数据库
    @classmethod
    @asynccontextmanager
    async def connect(cls) -> AsyncIterator[Database]:
        with logfire.span("connect to DB"):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect() -> MySQLConnection:
        con = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", "1234"),
            database=os.getenv("MYSQL_DATABASE", "chat_app"),
            charset="utf8mb4",
        )
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INT AUTO_INCREMENT PRIMARY KEY, message_list TEXT);"
        )
        con.commit()
        return con

    # 异步方法，用于添加消息
    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            "INSERT INTO messages (message_list) VALUES (%s);",
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    # 异步方法，用于获取消息
    async def get_messages(self) -> list[ModelMessage]:
        # 异步执行
        c = await self._asyncify(
            self._execute, "SELECT message_list FROM messages order by id"
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:  # 遍历行
            messages.extend(
                ModelMessagesTypeAdapter.validate_json(row[0])
            )  # 验证JSON并添加到消息列表
        return messages

    # 方法，用于执行SQL语句（MySQL 占位符为 %s）
    def _execute(self, sql: LiteralString, *args: Any, commit: bool = False) -> Any:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    # 异步方法，用于异步执行
    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


# 主函数
if __name__ == "__main__":
    import uvicorn

    # 运行FastAPI应用
    uvicorn.run("chat_app:app", reload=True, reload_dirs=[str(THIS_DIR)])
