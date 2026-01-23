from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 1. 加载文档
loader = DirectoryLoader(
    "Data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

# 2. 切分文本（推荐）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)

# 3. Embedding（⚠️ 必须用专用模型）
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b"
)

# 4. 构建向量库
docsearch = Chroma.from_documents(
    split_docs,
    embedding=embeddings
)

# 5. LLM（回答问题）
llm = OllamaLLM(
    model="qwen3:4b"
)

# 6. RAG QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True
)

# 7. 查询
result = qa.invoke({"query": "dify是什么"})
print(result["result"])

