# chain = prompt | model | parser | model | parser
# 在前面我们完成了这样的需求去构建多模型链，不过这种做法并不标准
# 因为：上一个模型的输出，没有被处理就输入下一个模型。
# 正常情况下我们应该有如下处理逻辑：
# invoke｜stream 初始输入 ➔ 提示词模板 ➔ 模型 ➔ 数据处理 ➔ 提示词模板 ➔ 模型 ➔ 解析器 ➔ 结果
# 即：上一个模型的输出结果，应该作为提示词模版的输入，构建下一个提示词，用来二次调用模型。


# 我们需要完成：
# 将模型输出的AIMessage➔转为字典➔注入第二个提示词模板中，形成新的提示词（PromptValue对象）
# StrOutputParser不满足（AIMessage ➔ Str）
# 更换JsonOutputParser（AIMessage ➔ Dict(JSON)）

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装到JSON格式返回给我，"
    "要求key是name，value就是起的名字。请严格遵守格式要求"
)

second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

# chain = first_prompt | model | json_parser | second_prompt | model | str_parser
chain = first_prompt | model | json_parser | second_prompt | model 

# res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
# print(res)
res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res.content)
print(type(res))