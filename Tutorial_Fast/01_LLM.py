from langchain_community.llms.tongyi import Tongyi

llm = Tongyi(model="qwen-max")
res = llm.invoke("你好，今天天气怎么样？")
print(res)
