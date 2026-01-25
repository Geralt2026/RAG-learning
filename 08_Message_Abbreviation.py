from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")

# 通过2元元组封装信息；
# 第一个元素为角色
# 字符串：system/human/ai
# 第二个元素为内容
messages = [
    ("system", "你是一个来自边塞的诗人"),
    ("human", "给我写一首唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "根据你上一首的格式，再来一首")
]

for chunk in model.stream(input=messages):
    print(chunk.content, end="", flush=True)

# 区别和优势在于，使用类对象的方式， 如下：
'''
messages = [
SystemMessage(content="内容…"),
HumanMessage(content "内容…"),
AIMessage(content= "内容…"),
]

是静态的，一步到位
直接就得到了Message类的类对象
'''

# 消息简写是动态的，需要在运行时，由LangChain内部机制转换为Message类对象
# 好处就在于，简写形式避免导包、写起来更简单，更重要的是支持：
'''
messages = [
(“system”, “今天的天气是{weather}”),
(“human”, “我的名字是：{name}”),
(“ai”, “欢迎{lastname}先生"),
]

由于是动态，需要转换步骤
所以简写形式支持内部填充{变量}占位
可在运行时填充具体值
'''
