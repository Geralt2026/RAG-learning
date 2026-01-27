'''
todo：修复在浏览器里的输出报错问题
'''

from __future__ import annotations as _annotations

import inspect
import json
import sys
from pathlib import Path

from httpx import AsyncClient
import logfire
from pydantic import BaseModel

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

# 保证从任意目录运行都能找到同目录下的 WeatherAgent 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from WeatherAgent import weather_agent, Deps

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `pip install gradio`. You must use python>=3.10.'
    ) from e

logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()  # 配置 PydanticAI 的日志

TOOL_TO_DISPLAY_NAME = {
    'get_lat_lng': 'Geocoding API',
    'get_weather': 'Weather API',
    'get_my_location': 'IP 定位（我的位置）',
}

# 兼容不同 Gradio 版本：
# - 3.x：无 type 参数，默认 (user, bot) 元组，不传 type
# - 4.x/5.x：有 type；若用元组格式必须传 type='tuples'，否则会按 messages 校验报 "Data incompatible with messages format"
def _use_messages_type():
    try:
        sig = inspect.signature(gr.Chatbot)
        return "type" in sig.parameters
    except Exception:
        return False

client = AsyncClient()
deps = Deps(client=client)


async def stream_from_agent(prompt: str, chatbot: list, past_messages: list):
    if _use_messages_type():
        chatbot.append({'role': 'user', 'content': prompt})
    else:
        chatbot.append([prompt, ""])
    yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()

    async with weather_agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        if _use_messages_type():
            tool_call_ids = {}  # tool_call_id -> 对应 message 在 chatbot 中的下标
            for message in result.new_messages():
                for call in message.parts:
                    if isinstance(call, ToolCallPart):
                        call_args = call.args_as_json_str()
                        display_name = TOOL_TO_DISPLAY_NAME.get(call.tool_name, call.tool_name)
                        # 只传 role/content，不传 metadata，避免 Gradio 报 "Data incompatible with messages format"
                        gr_message = {'role': 'assistant', 'content': f'🛠️ Using {display_name}\nParameters: {call_args}'}
                        chatbot.append(gr_message)
                        if call.tool_call_id is not None:
                            tool_call_ids[call.tool_call_id] = len(chatbot) - 1
                    if isinstance(call, ToolReturnPart):
                        idx = tool_call_ids.get(call.tool_call_id)
                        if idx is not None:
                            json_content = call.content.model_dump_json() if isinstance(call.content, BaseModel) else json.dumps(call.content)
                            chatbot[idx]['content'] += f'\nOutput: {json_content}'
                    yield gr.skip(), chatbot, gr.skip()
            chatbot.append({'role': 'assistant', 'content': ''})
            async for message in result.stream_text():
                chatbot[-1]['content'] = message
                yield gr.skip(), chatbot, gr.skip()
        else:
            bot_parts = []
            for message in result.new_messages():
                for call in message.parts:
                    if isinstance(call, ToolCallPart):
                        call_args = call.args_as_json_str()
                        display_name = TOOL_TO_DISPLAY_NAME.get(call.tool_name, call.tool_name)
                        bot_parts.append(f"🛠️ Using {display_name}\nParameters: {call_args}")
                    if isinstance(call, ToolReturnPart):
                        json_content = call.content.model_dump_json() if isinstance(call.content, BaseModel) else json.dumps(call.content)
                        if bot_parts:
                            bot_parts[-1] += f"\nOutput: {json_content}"
                    yield gr.skip(), chatbot, gr.skip()
            full_bot = "\n\n".join(bot_parts) if bot_parts else ""
            async for message in result.stream_text():
                full_bot = (full_bot + "\n\n" if full_bot else "") + message
                chatbot[-1] = [prompt, full_bot]
                yield gr.skip(), chatbot, gr.skip()

        past_messages = result.all_messages()
        yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    if _use_messages_type():
        previous_prompt = chatbot[retry_data.index].get('content', '') if isinstance(chatbot[retry_data.index], dict) else chatbot[retry_data.index][0]
    else:
        previous_prompt = chatbot[retry_data.index][0]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    prev = chatbot[undo_data.index]
    prev_content = prev.get('content', prev[0]) if isinstance(prev, dict) else prev[0]
    return prev_content, new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value.get('text', message.value) if isinstance(message.value, dict) else str(message.value)


with gr.Blocks() as demo:
    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
    <img src="https://ai.pydantic.org.cn/img/logo-white.svg" style="max-width: 200px; height: auto">
    <div>
        <h1 style="margin: 0 0 1rem 0">Weather Assistant</h1>
        <h3 style="margin: 0 0 0.5rem 0">
            This assistant answer your weather questions.
        </h3>
    </div>
</div>
"""
    )
    past_messages = gr.State([])
    chatbot_kwargs = {'label': 'Weather Assistant'}
    if _use_messages_type():
        chatbot_kwargs['type'] = 'messages'
        chatbot_kwargs['avatar_images'] = (None, 'https://ai.pydantic.org.cn/img/logo-white.svg')
        chatbot_kwargs['examples'] = [{'text': 'What is the weather like in Miami?'}, {'text': 'What is the weather like in London?'}]
    else:
        # 使用 (user, bot) 元组格式。Gradio 3.x 无 type 不传；4.x/5.x 若有 type 需传 type='tuples' 否则会按 messages 校验报错
        try:
            sig = inspect.signature(gr.Chatbot)
            if "type" in sig.parameters:
                chatbot_kwargs["type"] = "tuples"
        except Exception:
            pass
        # 3.x 的 examples 格式不同，不传以免 example.get("files") 报错
    chatbot = gr.Chatbot(**chatbot_kwargs)
    with gr.Row():
        prompt = gr.Textbox(
            lines=1,
            show_label=False,
            placeholder='What is the weather like in New York City?',
        )
    prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages],
        outputs=[prompt, chatbot, past_messages],
    )
    # 以下为 Gradio 4.x 的 Chatbot 能力，3.x 可能没有，用 try 避免报错
    try:
        chatbot.example_select(select_data, None, [prompt])
    except AttributeError:
        pass
    try:
        chatbot.retry(handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages])
    except AttributeError:
        pass
    try:
        chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])
    except AttributeError:
        pass


if __name__ == '__main__':
    try:
        logfire.instrument_httpx(client, capture_all=True)
    except Exception:
        pass  # 部分 logfire 版本无此 API 或无 client 时忽略
    demo.launch()