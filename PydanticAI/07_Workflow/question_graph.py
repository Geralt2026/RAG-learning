"""一个用于提问和评估问题的图表示例

Run with:

    uv run -m pydantic_ai_examples.question_graph
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import httpx
import logfire
from pydantic import BaseModel

from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_graph.persistence.file import FileStatePersistence
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

# 创建供 OpenAI 使用的 httpx 客户端并做 HTTP 观测，再传给 AsyncOpenAI
http_client = httpx.AsyncClient()
logfire.instrument_httpx(http_client, capture_all=True)
    
# 创建OpenAI客户端
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    http_client=http_client,
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))
ask_agent = Agent(model, output_type=str)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        result = await ask_agent.run(
            '请提出一个简单的问题，并给出正确答案。',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """答案是否正确"""
    comment: str
    """对答案的评论，如果答案错误，则批评用户。"""


evaluate_agent = Agent(
    model,
    output_type=EvaluationOutput,
    system_prompt='根据问题和答案，评估答案是否正确。',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.all_messages()
        if result.output.correct:
            return End(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)


async def run_as_continuous():
    state = QuestionState()
    node = Ask()
    end = await question_graph.run(node, state=state)
    print('END:', end.output)


async def run_as_cli(answer: str | None):
    persistence = FileStatePersistence(Path('question_graph.json'))
    persistence.set_graph_types(question_graph)

    if snapshot := await persistence.load_next():
        state = snapshot.state
        assert answer is not None, (
            '答案是必需的，使用 "uv run -m pydantic_ai_examples.question_graph cli <answer>"'
        )
        node = Evaluate(answer)
    else:
        state = QuestionState()
        node = Ask()
    # debug(state, node)

    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()
            if isinstance(node, End):
                print('END:', node.data)
                history = await persistence.load_all()
                print('history:', '\n'.join(str(e.node) for e in history), sep='\n')
                print('Finished!')
                break
            elif isinstance(node, Answer):
                print(node.question)
                break
            # otherwise just continue


if __name__ == '__main__':
    import asyncio
    import sys

    try:
        sub_command = sys.argv[1]
        assert sub_command in ('continuous', 'cli', 'mermaid')
    except (IndexError, AssertionError):
        print(
            'Usage:\n'
            '  uv run -m pydantic_ai_examples.question_graph mermaid\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph continuous\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph cli [answer]',
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == 'mermaid':
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == 'continuous':
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))