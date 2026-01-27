"""使用 PydanticAI 和多个工具的示例，LLM 需要依次调用这些工具来回答一个问题。

在这个例子中，我们的想法是一个“天气”代理——用户可以询问多个城市的天气，
代理将使用 `get_lat_lng` 或 `get_my_location` 获取经纬度，再使用 `get_weather` 获取天气。

- 问「北京/广州的天气」：Agent 会调用 get_lat_lng("北京") 再 get_weather(lat, lng)
- 问「我所在地/本地的天气」：Agent 会先调用 get_my_location（按当前 IP 解析所在地），再 get_weather(lat, lng)

配置说明：
1. 设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key，必需）
2. 可选：WEATHER_API_KEY（tomorrow.io）、LOCATION_API_KEY（geocode.maps.co）
3. 获取「用户所在地」使用 ip-api.com，无需配置，按当前请求 IP 解析

运行方式：
    python 03_WeatherAgent.py
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# =====================配置区=======================
# 配置日志
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()  # 配置 PydanticAI 的日志

# 获取 API Keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
LOCATION_API_KEY = os.getenv("LOCATION_API_KEY")
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

# 创建自定义的 OpenAI 客户端，使用阿里云百炼的兼容端点
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key
)

# 创建模型，使用阿里云百炼的 qwen-max
model = OpenAIChatModel(
    "qwen-max",
    provider=OpenAIProvider(openai_client=client)
)

print(f"✅ 使用模型: qwen-max (阿里云百炼)")


# =====================数据模型区=======================
# 定义依赖项（Dependencies）
# 依赖项用于在工具之间共享资源，如 HTTP 客户端
@dataclass
class Deps:
    client: AsyncClient  # 异步 HTTP 客户端，用于工具中发起网络请求


# 定义 LatLng 模型（经纬度坐标）
# 这是一个 Pydantic 模型，用于类型安全和数据验证
class LatLng(BaseModel):
    lat: float  # 纬度
    lng: float  # 经度


# 用户所在地信息（经纬度 + 城市名，用于「我所在地」场景）
class MyLocation(BaseModel):
    lat: float
    lng: float
    city: str = ""
    country: str = ""


# =====================Agent区=======================
# 创建天气代理 Agent
# 使用上面创建的 model（阿里云百炼 qwen-max）
# deps_type 指定依赖类型，用于在工具中访问共享资源（如 HTTP 客户端）
# retries 指定重试次数，当工具调用失败时会自动重试
weather_agent = Agent(
    model,  # 使用统一配置的模型实例
    instructions=(
        "Be concise, reply with one sentence. "
        "当用户询问「我所在地」「本地」「当前位置」「我这里的天气」时，"
        "必须先调用 get_my_location 获取当前用户所在位置的经纬度，再调用 get_weather 传入该经纬度获取天气。"
    ),
    deps_type=Deps,  # 依赖类型，工具可以通过 ctx.deps 访问
    retries=2,  # 重试次数
)


# =====================工具区=======================
async def _fetch_my_location(client: AsyncClient) -> MyLocation:
    """根据当前 IP 获取用户所在地经纬度与城市信息（供工具与 main 复用）"""
    # ip-api.com 免费接口，无需 API Key，不传 query 则使用当前请求 IP
    # 文档：http://ip-api.com/docs/api:json  注意免费版仅支持 http
    r = await client.get(
        "http://ip-api.com/json/",
        params={"fields": "status,message,lat,lon,city,country", "lang": "zh-CN"},
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise ValueError(data.get("message", "无法解析当前 IP 所在地"))
    return MyLocation(
        lat=float(data["lat"]),
        lng=float(data["lon"]),
        city=data.get("city", ""),
        country=data.get("country", ""),
    )


@weather_agent.tool
async def get_my_location(ctx: RunContext[Deps]) -> MyLocation:
    """获取当前用户所在地的经纬度和城市信息
    
    根据当前请求的 IP 解析用户大致所在地（城市、国家、经纬度）。
    当用户问「我所在地的天气」「本地的天气」「当前位置天气」时，应先调用此工具再调用 get_weather。
    """
    return await _fetch_my_location(ctx.deps.client)


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """获取位置的经纬度
    
    这个工具用于根据位置描述（如城市名称）获取对应的经纬度坐标。
    Agent 会自动调用此工具来获取位置信息。

    Args:
        ctx: 运行上下文，包含依赖项（如 HTTP 客户端）
        location_description: 位置描述，例如 "北京"、"广州" 等城市名称

    Returns:
        LatLng: 包含纬度和经度的 Pydantic 模型对象
    """
    if not LOCATION_API_KEY:
        raise ValueError("请设置环境变量 LOCATION_API_KEY（geocode.maps.co API Key）")
    
    # 使用 geocode.maps.co 的正向地理编码 API（从地址到坐标）
    # 文档：https://geocode.maps.co/docs/endpoints/
    r = await ctx.deps.client.get(
        "https://geocode.maps.co/search",
        params={
            "q": location_description,  # 必需：搜索查询
            "api_key": LOCATION_API_KEY,  # 必需：API Key
            "limit": 1,  # 只返回第一个结果
            "format": "json",  # 返回 JSON 格式
        },
    )
    r.raise_for_status()
    
    # 解析响应：geocode.maps.co/search 返回一个数组
    results = r.json()
    if not results or len(results) == 0:
        raise ValueError(f"未找到位置: {location_description}")
    
    # 获取第一个结果
    first_result = results[0]
    return LatLng(
        lat=float(first_result["lat"]),
        lng=float(first_result["lon"])
    )


async def _fetch_weather(client: AsyncClient, lat: float, lng: float) -> dict[str, Any]:
    """根据经纬度获取天气（供工具与 main 复用）"""
    if not WEATHER_API_KEY:
        raise ValueError("请设置环境变量 WEATHER_API_KEY（tomorrow.io API Key）")
    tasks = [
        client.get(
            "https://api.tomorrow.io/v4/timelines",
            params={
                "location": f"{lat},{lng}",
                "fields": "temperature",
                "units": "metric",
                "timesteps": "1h",
                "apikey": WEATHER_API_KEY,
            },
        ),
    ]
    if LOCATION_API_KEY:
        tasks.append(
            client.get(
                "https://geocode.maps.co/reverse",
                params={"lat": lat, "lon": lng, "api_key": LOCATION_API_KEY, "format": "json"},
            )
        )
    responses = await asyncio.gather(*tasks)
    temp_response = responses[0]
    descr_response = responses[1] if len(responses) > 1 else None
    temp_response.raise_for_status()
    weather_data = temp_response.json()
    try:
        temperature = weather_data["data"]["timelines"][0]["intervals"][0]["values"]["temperature"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"无法解析天气数据: {e}")
    description = "未知位置"
    if descr_response:
        descr_response.raise_for_status()
        description = descr_response.json().get("display_name", "未知位置")
    return {"temperature": f"{temperature} °C", "description": description}


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """获取指定经纬度的天气信息
    
    这个工具用于根据经纬度坐标获取该位置的天气信息。
    当用户问「我所在地的天气」时，应先调用 get_my_location 得到经纬度，再调用本工具。

    Args:
        ctx: 运行上下文，包含依赖项（如 HTTP 客户端）
        lat: 纬度（latitude）
        lng: 经度（longitude）

    Returns:
        dict: 包含温度（temperature）和描述（description）的字典
    """
    return await _fetch_weather(ctx.deps.client, lat, lng)


# =====================主函数=======================
async def main():
    """主函数：先获取用户所在地经纬度与天气，再让 Agent 回答「我所在地的天气」"""
    async with AsyncClient() as http_client:
        logfire.instrument_httpx(http_client, capture_all=True)
        deps = Deps(client=http_client)

        # 1. 获取用户所在地经纬度（基于当前 IP）
        try:
            my_loc = await _fetch_my_location(http_client)
            print(f"📍 当前解析到的位置: {my_loc.city or '(未知城市)'} {my_loc.country or ''} ({my_loc.lat:.4f}, {my_loc.lng:.4f})")
        except Exception as e:
            print(f"⚠️ 无法获取所在地: {e}")
            my_loc = None

        # 2. 若有天气 API Key，可直接查该地天气（不经过 Agent）
        if my_loc and WEATHER_API_KEY:
            try:
                weather = await _fetch_weather(http_client, my_loc.lat, my_loc.lng)
                print(f"🌤️ 当地天气: {weather['temperature']} | {weather['description']}")
            except Exception as e:
                print(f"⚠️ 获取天气失败: {e}")

        # 3. 用 Agent 回答「我所在地的天气」：Agent 会先调 get_my_location 再调 get_weather
        result = await weather_agent.run(
            "我所在地的天气如何？",
            deps=deps,
        )
        print("\n📊 Agent 响应:")
        print(result.output)
        usage = result.usage()
        if usage:
            print(f"\n📈 使用情况: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
