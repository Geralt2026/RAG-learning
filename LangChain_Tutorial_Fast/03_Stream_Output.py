# 如果需要流式输出结果，需要将模型的invoke方法改为stream方法即可。
# • invoke方法：一次型返回完整结果
# • stream方法：逐段返回结果，流式输出

# langchain_community
from langchain_community.llms.tongyi import Tongyi

# 不用qwen3-max，因为qwen3-max是聊天模型，qwen-max是大语言模型
model = Tongyi(model="qwen-max")

# 调用invoke向模型提问
res = model.stream(input="你是谁呀能做什么？")

for chunk in res:
    print(chunk, end="", flush=True)
