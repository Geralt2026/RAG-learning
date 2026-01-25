from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

#初始化模型
chat = ChatTongyi(model="qwen3-max")

#准备消息list
messages = [
    HumanMessage(content="给我写一首唐诗")
]

#流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)