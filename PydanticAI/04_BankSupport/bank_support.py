"""Small but complete example of using Pydantic AI to build a support agent for a bank.

Run with:

    uv run -m pydantic_ai_examples.bank_support
"""

from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
import logfire

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


# 创建数据库连接类
class DatabaseConn:
    """这是一个假的数据库，用于示例目的。
    在现实中，你会连接到一个外部数据库
    (例如 PostgreSQL) 来获取客户信息。
    """

    # 类方法，用于获取客户姓名
    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return "John"

    # 类方法，用于获取客户余额
    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            if include_pending:
                return 123.45
            else:
                return 100.00
        else:
            raise ValueError("Customer not found")


# 创建支持依赖类
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# 创建支持输出类
class SupportOutput(BaseModel):
    support_advice: str
    """Advice returned to the customer"""
    block_card: bool
    """Whether to block their card or not"""
    risk: int
    """Risk level of query"""


# 创建支持代理
support_agent = Agent(
    model,
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        "你是我们银行的客服人员，为客户提供支持并判断他们查询的风险等级。"
        "回复时使用客户的姓名。"
    ),
)


# 创建支持指令
@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"客户的姓名是 {customer_name!r}"


# 创建支持工具
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """返回客户的当前账户余额。"""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f"客户的当前账户余额是 ${balance:.2f}"


# 主函数
if __name__ == "__main__":
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = support_agent.run_sync("我的余额是多少？", deps=deps)
    print(result.output)
    """
    support_advice='你好，张三，你的当前账户余额，包括待处理交易，是 $123.45。' block_card=False risk=1
    """

    result = support_agent.run_sync("我刚刚丢失了我的卡片！", deps=deps)
    print(result.output)
    """
    support_advice="很抱歉听到这个消息，张三。我们暂时封锁您的卡片以防止未经授权的交易。" block_card=True risk=8
    """
