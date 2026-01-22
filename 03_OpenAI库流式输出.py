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
        {"role": "system", "content": "你是一个Python编程专家。话非常多"}, 
        {"role": "assistant", "content": "我是一个Python编程专家。而且话非常多，请问有什么可以帮助您的吗？"},
        {"role": "user", "content": "for循环输出1到5的数字"}
    ],
    stream=True
)

# 3. 处理结果
for chunk in response:
    print(chunk.choices[0].delta.content, 
          end=" ", # 每一段之间以空格分隔
          flush=True# 立即输出，而不是等待缓冲区满
          )
