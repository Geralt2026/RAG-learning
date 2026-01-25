import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# 1. 获取Client对象，OpenAI类对象
client = OpenAI(  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 2. 调用模型
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个AI助理，回答很简洁"}, 
        {"role": "user", "content": "小明有2条宠物狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小明有3条宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "一共有几个宠物？"}
    ],
    stream=True
)

# 3. 处理结果
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
