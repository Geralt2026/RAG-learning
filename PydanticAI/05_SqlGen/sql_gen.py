"""此示例演示如何使用 Pydantic AI 根据用户输入生成 SQL 查询。

MySQL 需事先启动，并通过环境变量配置：
MYSQL_HOST、MYSQL_PORT、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE（默认 sql_gen）。

Run with:

    pip install aiomysql
    python sql_gen.py
    python sql_gen.py "show me logs from yesterday, with level 'error'"
"""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypeAlias

import aiomysql
import logfire
from annotated_types import MinLen
from devtools import debug
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

# 创建OpenAI客户端
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))

# 配置日志
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

# 创建数据库表结构（MySQL 语法）
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    created_at DATETIME(6),
    start_timestamp DATETIME(6),
    end_timestamp DATETIME(6),
    trace_id TEXT,
    span_id TEXT,
    parent_span_id TEXT,
    level ENUM('debug','info','warning','error','critical'),
    span_name TEXT,
    message TEXT,
    attributes_json_schema TEXT,
    attributes JSON,
    tags JSON,
    is_exception BOOLEAN,
    otel_status_message TEXT,
    service_name TEXT
);
"""
# 创建SQL查询示例（MySQL 语法）
SQL_EXAMPLES = [
    {
        'request': '显示所有foobar为false的记录',
        'response': "SELECT * FROM records WHERE JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.foobar')) = 'false'",
    },
    {
        'request': '显示包含属性foobar的记录',
        'response': "SELECT * FROM records WHERE JSON_CONTAINS_PATH(attributes, 'one', '$.foobar')",
    },
    {
        'request': '显示昨天的记录',
        'response': "SELECT * FROM records WHERE DATE(start_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)",
    },
    {
        'request': '显示包含标签foobar的错误记录',
        'response': "SELECT * FROM records WHERE level = 'error' AND JSON_CONTAINS(tags, '\"foobar\"')",
    },
]


# 创建依赖类
@dataclass
class Deps:
    conn: aiomysql.Connection


# 创建成功响应类
class Success(BaseModel):
    """当SQL可以成功生成时的响应。"""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        '',
        description='SQL查询的解释，作为markdown',
    )


# 创建无效请求响应类
class InvalidRequest(BaseModel):
    """用户输入没有足够信息生成SQL时的响应。"""

    error_message: str


# 创建响应类型别名
Response: TypeAlias = Success | InvalidRequest


# 创建Agent
agent = Agent[Deps, Response](
    model,
    output_type=Response,  # type: ignore
    deps_type=Deps,
)


# 创建系统提示
@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
给定以下 MySQL 表中的记录，你的任务是
编写一个适合用户请求的 SQL 查询。

数据库表结构:

{DB_SCHEMA}

今天的日期 = {date.today()}

SQL 查询示例（MySQL 语法，使用 JSON 函数等）:
{format_as_xml(SQL_EXAMPLES)}
"""


# 创建输出验证器
@agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output

    # 有时LLM会添加不必要的反斜杠到SQL
    output.sql_query = output.sql_query.replace('\\', '')
    if not output.sql_query.upper().strip().startswith('SELECT'):
        raise ModelRetry('请创建一个SELECT查询')

    try:
        async with ctx.deps.conn.cursor() as cur:
            await cur.execute(f'EXPLAIN {output.sql_query}')
    except aiomysql.Error as e:
        raise ModelRetry(f'无效查询: {e}') from e
    else:
        return output


# 主函数
async def main():
    if len(sys.argv) == 1:
        prompt = '显示昨天的错误记录'
    else:
        prompt = sys.argv[1]

    async with database_connect() as conn:
        deps = Deps(conn)
        result = await agent.run(prompt, deps=deps)
    debug(result.output)


# 创建数据库连接
@asynccontextmanager
async def database_connect() -> AsyncGenerator[Any, None]:
    host = os.getenv('MYSQL_HOST', '127.0.0.1')
    port = int(os.getenv('MYSQL_PORT', '3306'))
    user = os.getenv('MYSQL_USER', 'root')
    password = os.getenv('MYSQL_PASSWORD', '1234')
    database = os.getenv('MYSQL_DATABASE', 'sql_gen')

    with logfire.span('检查并创建数据库'):
        conn = await aiomysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute(f'CREATE DATABASE IF NOT EXISTS `{database}`')
                await conn.commit()
        finally:
            conn.close()

    conn = await aiomysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        db=database,
        charset='utf8mb4',
    )
    try:
        with logfire.span('创建表结构'):
            async with conn.cursor() as cur:
                await cur.execute(DB_SCHEMA)
                await conn.commit()
        yield conn
    finally:
        conn.close()


if __name__ == '__main__':
    asyncio.run(main())
