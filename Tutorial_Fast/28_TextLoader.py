from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "xxx.txt",
    encoding="utf-8"
)

docs = loader.load()
print(docs)
print(len(docs))