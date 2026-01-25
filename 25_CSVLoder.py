from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./xxx.csv",
    csv_args={
        "delimiter": ",",   # 指定分隔符
        "quotechar": '"',   # 指定字符串的引号包裹
        "fieldnames": ["name", "age", "gender"], # 字段列表（无表头使用，有表头勿用会读取首行做为数据）
    },
)

# 一次性加载全部文档
documents = loader.load()

# 对于大数据集，分段返回文档
for chunk in loader.lazy_load():
    print(chunk)