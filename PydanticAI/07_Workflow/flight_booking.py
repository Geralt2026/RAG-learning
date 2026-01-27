"""一个多Agent流程的示例，其中一个Agent将工作委托给另一个Agent，然后将控制权移交给第三个Agent。

在这个场景中，一组Agent协同工作，为用户找到航班。
"""

import datetime
from dataclasses import dataclass
import os
from typing import Literal

import httpx
import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelRetry, RunContext, RunUsage, UsageLimits
from pydantic_ai.messages import ModelMessage
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


class FlightDetails(BaseModel):
    """最合适的航班详情。"""

    flight_number: str
    price: int
    origin: str = Field(description='三字母机场代码')
    destination: str = Field(description='三字母机场代码')
    date: datetime.date


class NoFlightFound(BaseModel):
    """当没有找到有效的航班时。"""


@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date


# 这个Agent负责控制对话的流程。
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    model,
    output_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        '你的任务是找到给定日期用户最便宜的航班。 '
    ),
)


# 这个Agent负责从网页文本中提取航班详情。
extraction_agent = Agent(
    model,
    output_type=list[FlightDetails],
    system_prompt='从给定的文本中提取所有航班的详情。',
)


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """获取所有航班的详情。"""
    # we pass the usage to the search agent so requests within this agent are counted
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.output))
    return result.output


@search_agent.output_validator
async def validate_output(
    ctx: RunContext[Deps], output: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """流程验证，确保航班符合约束条件。"""
    if isinstance(output, NoFlightFound):
        return output

    errors: list[str] = []
    if output.origin != ctx.deps.req_origin:
        errors.append(
            f'航班应该有出发地 {ctx.deps.req_origin}，不是 {output.origin}'
        )
    if output.destination != ctx.deps.req_destination:
        errors.append(
            f'航班应该有目的地 {ctx.deps.req_destination}，不是 {output.destination}'
        )
    if output.date != ctx.deps.req_date:
        errors.append(f'航班应该在 {ctx.deps.req_date}，不是 {output.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return output


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']


class Failed(BaseModel):
    """无法提取座位选择。"""


# 这个Agent负责提取用户的座位选择
seat_preference_agent = Agent[None, SeatPreference | Failed](
    model,
    output_type=SeatPreference | Failed,
    system_prompt=(
        "提取用户的座位偏好。 "
        '座位A和F是靠窗座位。 '
        '第1排是前排，有额外的腿部空间。 '
        '第14排和第20排也有额外的腿部空间。 '
    ),
)


# 在现实中，这将从预订网站下载，可能使用另一个Agent来导航网站
flights_web_page = """
1. 航班 SFO-AK123
- 价格: ¥350
- 出发地: 旧金山国际机场 (SFO)
- 目的地: 安克雷奇国际机场 (ANC)
- 日期: 2025年1月10日

2. 航班 SFO-AK456
- 价格: ¥370
- 出发地: 旧金山国际机场 (SFO)
- 目的地: 费尔班克斯国际机场 (FAI)
- 日期: 2025年1月10日

3. 航班 SFO-AK789
- 价格: ¥400
- 出发地: 旧金山国际机场 (SFO)
- 目的地: 朱诺国际机场 (JNU)
- 日期: 2025年1月20日

4. 航班 NYC-LA101
- 价格: ¥250
- 出发地: 旧金山国际机场 (SFO)
- 目的地: 安克雷奇国际机场 (ANC)
- 日期: 2025年1月10日

5. 航班 CHI-MIA202
- 价格: ¥200
- 出发地: 芝加哥奥黑尔国际机场 (ORD)
- 目的地: 迈阿密国际机场 (MIA)
- 日期: 2025年1月12日

6. 航班 BOS-SEA303
- 价格: ¥120
- 出发地: 波士顿洛根国际机场 (BOS)
- 目的地: 安克雷奇国际机场 (ANC)
- 日期: 2025年1月12日

7. 航班 DFW-DEN404
- 价格: ¥150
- 出发地: 达拉斯/沃思堡国际机场 (DFW)
- 目的地: 丹佛国际机场 (DEN)
- 日期: 2025年1月10日

8. 航班 ATL-HOU505
- 价格: ¥180
- 出发地: 亚特兰大哈茨菲尔德-杰克逊国际机场 (ATL)
- 目的地: 乔治·布什国际机场 (IAH)
- 日期: 2025年1月10日
"""

# 限制这个应用程序可以向LLM发出的请求数量
usage_limits = UsageLimits(request_limit=15)


async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: RunUsage = RunUsage()
    # 运行Agent，直到找到满意的航班
    while True:
        result = await search_agent.run(
            f'找到从 {deps.req_origin} 到 {deps.req_destination} 在 {deps.req_date} 的航班',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, NoFlightFound):
            print('没有找到航班')
            break
        else:
            flight = result.output
            print(f'找到航班: {flight}')
            answer = Prompt.ask(
                '你想购买这个航班，还是继续搜索？ 请输入：购买/继续搜索',
                choices=['购买', '继续搜索', ''],
                show_choices=False,
            )
            if answer == '购买':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all_messages(
                    output_tool_return_content='请建议另一个航班'
                )


async def find_seat(usage: RunUsage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('你想坐哪个座位？ 请输入：1-30/A-F')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            print('无法理解座位偏好。请再试一次。')
            message_history = result.all_messages()


async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'购买航班 {flight_details=!r} {seat=!r}...')


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    