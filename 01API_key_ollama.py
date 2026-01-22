import os
from openai import OpenAI
# 1. 获取Client对象，OpenAI类对象
client = OpenAI(  
    base_url="http://localhost:11434/v1",
)
# 2. 调用模型
completion = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)