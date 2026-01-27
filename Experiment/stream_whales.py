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
model = OpenAIChatModel("qwen3-max", provider=OpenAIProvider(openai_client=client))

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
    console = Console()
    # 使用 Live 模式，并将 vertical_overflow 设为 visible
    with Live(console=console, refresh_per_second=10) as live:
        console.print('正在请求鲸鱼数据...', style='bold cyan')
        
        # 运行 Agent
        async with agent.run_stream('生成关于5种鲸鱼的信息。') as result:
            # 关键点：使用 stream_output(partial=True)
            # partial=True 允许在对象字段尚未完整时就产出中间状态
            async for whales in result.stream_output(partial=True):
                table = Table(
                    title='[bold blue]流式鲸鱼数据采集[/bold blue]',
                    caption='数据正在实时验证并填充...',
                    width=120,
                    show_header=True,
                    header_style="bold magenta"
                )
                table.add_column('ID', width=4)
                table.add_column('名称', width=15)
                table.add_column('长度 (m)', justify='right', width=12)
                table.add_column('重量 (kg)', justify='right', width=15)
                table.add_column('海域', width=15)
                table.add_column('描述', justify='left', ratio=1)

                for wid, whale in enumerate(whales, start=1):
                    # 安全地获取字段，如果字段尚不存在则显示 "..."
                    name = whale.get('name') or '[dim]读取中...[/dim]'
                    length = f"{whale.get('length'):.1f}" if whale.get('length') is not None else '...'
                    weight = f"{whale.get('weight'):,.0f}" if whale.get('weight') is not None else '...'
                    ocean = whale.get('ocean') or '...'
                    # 对长文本描述进行切片或流式处理显示
                    desc = whale.get('description') or '...'
                    
                    table.add_row(
                        str(wid),
                        name,
                        length,
                        weight,
                        ocean,
                        desc
                    )
                
                live.update(table)

    console.print('[bold green]✔ 数据采集完成！[/bold green]')


if __name__ == '__main__':
    import asyncio
    # 运行主函数
    asyncio.run(main())