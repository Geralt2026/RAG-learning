# RecursiveCharacterTextSplitter，递归字符文本分割器，主要用于按自然段落分割大文档
# 是LangChain官方推荐的默认字符分割器
# 它在保持上下文完整性和控制片段大小之间实现了良好平衡，开箱即用效果佳

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    ".\Data\测试文本_Python基础语法.txt",
    encoding="utf-8"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 每个片段的长度
    chunk_overlap=50, # 片段之间的重叠长度
    # 文本分段依据
    separators=["\n\n", "\n", "。", "，", "？", "！", "：", "；", "、", "|", " "],
    # 字符统计依据（函数）
    length_function=len
)

split_docs = splitter.split_documents(docs)
print(split_docs)
print(len(split_docs))