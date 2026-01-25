from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_template = PromptTemplate.from_template(
    "产品名称：{good_name}，核心卖点：{description}"
)

# 示例数据，list内嵌套字典
example_data = [
    {"good_name": "MacBook Pro", "description": "高效节能、性能强大"},
    {"good_name": "联想笔记本", "description": "畅玩游戏，丝滑流畅"},
]

# FewShot提示词模板对象
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="基于示例，抽取产品名称和核心卖点2个字段，我提供两个示例",
    suffix="基于示例告诉我：{raw_text}的产品名称和核心卖点是什么？",
    input_variables=["raw_text"],
)

# 获得最终的提示词
prompt_text = few_shot_prompt.invoke(
    input={"raw_text": "华为matebook，高清大屏，长效续航，你的好帮手"}
).to_string()
# print(prompt_text)

model = Tongyi(model="qwen-max")

print(model.invoke(input = prompt_text))
