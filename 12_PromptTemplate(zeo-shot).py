# 提示词优化在模型应用中非常重要，LangChain提供了PromptTemplate类，用来协助优化提示词。
# PromptTemplate表示提示词模板，可以构建一个自定义的基础提示词模板，支持变量的注入，最终生成所需的提示词。

#基于chain的写法
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 帮忙起名字，请简略回答。"
)

model = Tongyi(model="qwen-max")

# 生成链
chain = prompt_template | model
# 基于链，调用模型获取结果
res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
