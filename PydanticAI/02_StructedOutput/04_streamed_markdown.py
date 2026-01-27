"""此示例展示了如何从代理 (agent) 流式传输 Markdown，并使用 rich 库在终端中高亮输出。
运行方式：

    python 04_streamed_markdown.py
"""

import asyncio
import os
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import logfire
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text
from pydantic_ai import Agent

# 配置阿里云百炼的 OpenAI 兼容 API
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 创建模型
model = OpenAIChatModel("qwen3-max", provider=OpenAIProvider(openai_client=client))

# 配置日志
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

# 创建代理
agent = Agent(model)

# 模型列表，和相应的环境变量
models: list[tuple[OpenAIChatModel, str]] = [
    (model, "DASHSCOPE_API_KEY"),
]


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
    for model, env_var in models:
        # 如果环境变量存在
        if env_var in os.environ:
            console.log(f"Using model: {model}")
            # 创建实时输出
            with Live("", console=console, vertical_overflow="visible") as live:
                # 运行代理
                async with agent.run_stream(prompt, model=model) as result:
                    # 流式输出
                    async for message in result.stream_output():
                        # 更新实时输出
                        live.update(Markdown(message))
            # 打印使用情况
            console.log(result.usage())
        # 如果环境变量不存在
        else:
            # 打印错误信息
            console.log(f"{model} requires {env_var} to be set.")


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
