# 「将组件串联，上一个组件的输出作为下一个组件的输入」是LangChain 链（尤其是| 管道链）的核心工作原理，
# 这也是链式调用的核心价值：实现数据的自动化流转与组件的协同工作，如下。
# chain = prompt_template | model
# 核心前提：即Runnable子类对象才能入链（以及Callable、Mapping接口子类对象也可加入)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.base import RunnableSerializable

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗，无需额外输出")
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

model = ChatTongyi(model="qwen3-max")

chain: RunnableSerializable = chat_prompt_template | model
print(type(chain))

# Runnable接口，invoke执行
res = chain.invoke({"history": history_data})
print(res.content)

# Runnable接口, stream执行
for chunk in chain.stream({"history":history_data}):
    print(chunk.content, end="", flush=True)

# LangChain中链是一种将各个组件串联在一起，按顺序执行，前一个组件的输出作为下一个组件的输入。
