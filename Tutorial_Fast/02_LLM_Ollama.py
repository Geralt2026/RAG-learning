from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

# 通过invoke方法去调用模型
res = model.invoke("你好，今天天气怎么样？")
print(res)
