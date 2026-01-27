"""此示例展示了如何从代理 (agent) 流式传输 Markdown，并使用 rich 库在终端中高亮输出。
运行方式：

    python 04_streamed_markdown.py
"""

import asyncio
import os
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text
from pydantic_ai import Agent

ollama_model = OpenAIChatModel(
    model_name='qwen3:4b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

agent = Agent(ollama_model)



async def main():
    # 美化代码块
    prettier_code_blocks()
    # 创建控制台
    console = Console()
    # 提示词
    prompt = "告诉我Python的基本语法"
    # 打印提示词
    console.log(f"Asking: {prompt}...", style="cyan")
    # 遍历模型列表
    with Live("", console=console, vertical_overflow="visible") as live:
        async with agent.run_stream(prompt) as result:
            async for message in result.stream_text(): # 流式输出，注意和stream_output()的区别。
                # stream_text()返回的是字符串，stream_output()返回的是模型输出的结构化数据，每次给的是整段累计内容
                live.update(Markdown(message))


def prettier_code_blocks():
    """美化代码块"""
    # 创建简单代码块
    class SimpleCodeBlock(CodeBlock):
        """简单代码块"""
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            """渲染代码块"""
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style="dim")
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color="default",
                word_wrap=True,
            )
            yield Text(f"/{self.lexer_name}", style="dim")

    # 设置代码块元素
    Markdown.elements["fence"] = SimpleCodeBlock


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
