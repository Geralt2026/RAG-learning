import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# 1. 获取Client对象，OpenAI类对象
client = OpenAI(  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 2. 调用模型
response: ChatCompletion = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个Python编程专家。"}, 
        {"role": "assistant", "content": "我是一个Python编程专家。请问有什么可以帮助您的吗？"},
        {"role": "user", "content": "for循环输出1到5的数字"}
    ],
)

# 3. 处理结果
print(response.choices[0].message.content)
