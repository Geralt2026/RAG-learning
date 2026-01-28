from __future__ import annotations as _annotations

from dataclasses import dataclass, field
import asyncio
import os
from pydantic import BaseModel, EmailStr
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    model,
    output_type=Email,
    system_prompt="写一份欢迎邮件给我们的技术博客。",
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f"重写邮件给用户:\n"
                f"{format_as_xml(ctx.state.user)}\n"
                f"Feedback: {self.email_feedback}"
            )
        else:
            prompt = (
                f"写一份欢迎邮件给用户:\n"
                f"{format_as_xml(ctx.state.user)}"
            )

        console = Console()
        with Live("", console=console, refresh_per_second=8) as live:
            async with email_writer_agent.run_stream(
                prompt,
                message_history=ctx.state.write_agent_messages,
            ) as result:
                final_email = None
                async for email_partial in result.stream_output(debounce_by=0.05):
                    if isinstance(email_partial, dict):
                        subj = email_partial.get("subject", "") or ""
                        body = email_partial.get("body", "") or ""
                        final_email = Email(subject=subj, body=body)
                    else:
                        subj = getattr(email_partial, "subject", "") or ""
                        body = getattr(email_partial, "body", "") or ""
                        final_email = email_partial if isinstance(email_partial, Email) else Email(subject=subj, body=body)
                    live.update(
                        Panel(
                            f"[bold]主题:[/bold] {subj}\n\n[bold]内容:[/bold]\n{body}",
                            title="📧 正在生成邮件...",
                            border_style="blue",
                        )
                    )
                if final_email is None:
                    try:
                        final_email = await result.get_output()
                    except (AttributeError, TypeError):
                        raise RuntimeError("邮件生成未返回内容，请重试。") from None
                ctx.state.write_agent_messages += result.new_messages()

        return Feedback(final_email)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    model,
    output_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        "审查邮件并提供反馈，邮件必须参考用户的特定兴趣。"
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        # 先打印邮件给用户看
        print(f"\n📧 生成的邮件:")
        print(f"主题: {self.email.subject}")
        print(f"内容:\n{self.email.body}\n")

        # 等待用户反馈
        user_feedback = Prompt.ask(
            "请输入反馈（直接回车表示满意，或输入修改建议）",
            default="",
        )

        if user_feedback.strip():
            # 用户提供了反馈，需要重写
            return WriteEmail(email_feedback=user_feedback)
        else:
            # 用户满意，结束
            return End(self.email)


async def main():
    user = User(
        name="John Doe",
        email="john.joe@example.com",
        interests=["Haskel", "Lisp", "Fortran"],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    result = await feedback_graph.run(WriteEmail(), state=state)
    print(result.output)
    """
    Email(
        subject="欢迎来到我们的技术博客！",
        body="你好 John, 欢迎来到我们的技术博客！...",
    )
    """

if __name__ == "__main__":
    asyncio.run(main())