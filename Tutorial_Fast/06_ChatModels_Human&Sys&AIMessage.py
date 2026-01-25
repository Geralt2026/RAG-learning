from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

#初始化模型
chat = ChatTongyi(model="qwen3-max")

#准备消息list
messages = [
    SystemMessage(content="你是一个来自边塞的诗人"),
    HumanMessage(content="给我写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="根据你上一首的格式，再来一首")
]

#流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)
