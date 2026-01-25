from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_classic.chains.summarize import load_summarize_chain

# 1. 加载文档
loader = TextLoader(
    "Data/测试文本_Python基础语法.txt",
    encoding="utf-8"
)
documents = loader.load()
print(f"原始文档段数: {len(documents)}")

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)
print(f"切分后的文档块数: {len(split_docs)}")

# 3. 本地 Ollama
llm = Ollama(model="qwen3:4b")

# 4. 总结链
chain = load_summarize_chain(
    llm,
    chain_type="refine",
    verbose=True
)

# 5. 执行
result = chain.invoke(split_docs[:5])
print(result["output_text"])
