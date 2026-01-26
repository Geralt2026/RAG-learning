# 前文我们根据JsonOutputParser完成了多模型执行链条的构建
# 除了JsonOutputParser这类固定功能的解析器之外
# 我们也可以自己编写Lambda匿名函数来完成自定义逻辑的数据转换，想怎么转换就怎么转换，更自由
# 想要完成这个功能，可以基于RunnableLambda类实现

# RunnableLambda类是LangChain内置的，将普通函数等转换为Runnable接口实例，方便自定义函数加入chain

# 语法：RunnableLambda(函数对象或lambda匿名函数)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

model = ChatTongyi(model="qwen3-max")
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字，不要额外信息"
)

second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)


chain = first_prompt | model | my_func | second_prompt | model | str_parser
# 注意：此行可以直接写为：
# chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser
# 跳过RunnableLambda类，直接让函数加入链也是可以的
# 因为Runnable接口类在实现__or__的时候，支持Callable接口的实例
# 其本质是将函数自动转换为RunnableLambda

res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
print(type(res))
