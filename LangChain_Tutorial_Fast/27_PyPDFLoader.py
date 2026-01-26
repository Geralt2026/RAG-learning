from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="./Data/test.pdf",  # 文件路径必填
    mode="page", # 读取模式，可选page（按页面划分不同Document）和single（单个Document）
    password="123456",  # 文件密码
)

document = loader.load()
print(document)
