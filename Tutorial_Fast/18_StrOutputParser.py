from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

# 有如下代码，想要以第一次模型的输出结果，第二次去询问模型：
model = ChatTongyi(model="qwen3-max")

prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}, 刚生了{gender}，请起名，仅告知名字无需其它内容"
)

chain = prompt | model | model

res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res.content)

# 链的构建完全符合要求（参与的组件）
# 但是运行报错，错误的主要原因是： prompt的结果是PromptValue类型，输入给了model，model的输出结果是：AIMessage

# StrOutputParser是LangChain内置的简单字符串解析器
# 可以将AIMessage解析为简单的字符串，符合了模型invoke方法要求（可传入字符串，不接收AIMessage类型）
# 是Runnable接口的子类（可以加入链）
# parser = StrOutputParser()
# chain = prompt | model | parser | model
