# JSONLoader用于将JSON数据加载为Document类型对象。
# 使用JSONLoader需要额外安装： pip install jq
# jq是一个跨平台的json解析工具，LangChain底层对JSON的解析就是基于jq工具实现的
# 将JSON数据的信息抽取出来，封装为Document对象，抽取的时候依赖jq_schema语法
# jq_schema语法：

"""
例子：
{
    "name": "周杰轮",
    "age": 11,
    "hobby": ["唱", "跳", "RAP"],
    "other": {
        "addr": "深圳",
        "tel": "12332112321"
    }
}


.表示整个JSON对象（根）
[]表示数组
.name表示抽取周杰轮
.hobby表示抽取爱好数组
.hobby[1]或.hobby.[1]表示抽取跳
.other.addr表示抽取地址深圳
"""

from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="./Data/stus.json",  # 文件路径，必填
    jq_schema=".[].name",  # jq schema语法，必填
    text_content=False, # 抽取的是否是字符串，默认True
    json_lines=False,  # 是否是JsonLines文件（每一行都是JSON的文件），标准JSON数组应设为False  
)

document = loader.load()
print(document)
