"""关于鲸鱼的信息 — 一个流式结构化响应验证的示例。

此脚本从 qwen3-max 模型 流式传输关于鲸鱼的结构化响应，验证数据，并在接收数据的同时使用 rich 将其显示为动态表格。

运行方式：

    python 06_stream_whales.py
"""

import os
from typing import Annotated

import logfire
from pydantic import Field
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

# 'if-token-present' 表示如果没有配置 logfire，则不发送任何内容
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

# 创建OpenAI客户端
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))

# 定义鲸鱼模型
class Whale(TypedDict):
    name: str
    length: Annotated[
        float, Field(description='Average length of an adult whale in meters.')
    ]
    weight: NotRequired[
        Annotated[
            float,
            Field(description='Average weight of an adult whale in kilograms.', ge=50),
        ]
    ]
    ocean: NotRequired[str]
    description: NotRequired[Annotated[str, Field(description='Short Description')]]

# 创建Agent对象  Agent对象是PydanticAI的入口，用于执行任务  model是模型，output_type是输出类型
agent = Agent(model, output_type=list[Whale])

# 主函数
async def main():
    # 创建控制台
    console = Console()
    # 创建实时输出  \n 是换行符  * 36 是36行
    with Live('\n' * 36, console=console) as live:
        # 打印请求数据
        console.print('Requesting data...', style='cyan')
        # 运行Agent 执行任务
        async with agent.run_stream('生成关于5种鲸鱼的信息。') as result:
            # 打印响应数据
            console.print('Response:', style='green')
            # 流式输出
            async for whales in result.stream_output(debounce_by=0.01): # debounce_by=0.01 是去抖动，防止数据流过于频繁 
            # async for whales in result.stream_output(debounce_by=None): # debounce=None 表示不加 debounce，一有可解析的局部结果就 yield，便于流式填充 
                # 创建表格
                table = Table(
                    title='鲸鱼种类',
                    caption='流式结构化响应',
                    width=120,
                )
                table.add_column('ID', justify='right')
                table.add_column('Name')
                table.add_column('Avg. Length (m)', justify='right')
                table.add_column('Avg. Weight (kg)', justify='right')
                table.add_column('Ocean')
                table.add_column('Description', justify='right')

                for wid, whale in enumerate(whales, start=1): # 遍历鲸鱼列表  wid是编号，whale是鲸鱼
                    table.add_row( # 添加行
                        str(wid),
                        whale['name'],
                        f'{whale["length"]:0.0f}',
                        f'{w:0.0f}' if (w := whale.get('weight')) else '…',
                        whale.get('ocean') or '…',
                        whale.get('description') or '…',
                    )
                live.update(table) # 更新实时输出


if __name__ == '__main__':
    import asyncio
    # 运行主函数
    asyncio.run(main())