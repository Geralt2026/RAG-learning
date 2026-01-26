"""
知识库
"""

from datetime import datetime
import hashlib
import os
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过"""
    if not os.path.exists(config.md5_path):
        # 文件不存在，认为没有处理过
        open(config.md5_path, "w", encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()  # 去掉字符串前后的空格和回车
            if line and line == md5_str:  # 跳过空行
                return True  # 已经处理过
        return False


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串转化为md5字符串"""
    str_bytes = input_str.encode(encoding)  # 将字符串编码为bytes
    md5 = hashlib.md5()  # 创建一个md5对象
    md5.update(str_bytes)  # 更新md5对象
    md5_str = md5.hexdigest()  # md5字符串
    return md5_str


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果向量库保存路径不存在，则创建，存在则跳过
        os.makedirs(config.chroma_persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.chroma_collection_name,  # 向量库名称
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v4"
            ),  # 嵌入模型
            persist_directory=config.chroma_persist_directory,  # 向量库保存路径
        )  # 初始化Chroma向量库

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 每个片段的长度
            chunk_overlap=config.chunk_overlap,  # 片段之间的重叠长度
            separators=config.separators,  # 分隔符
            length_function=len,  # 字符统计
        )  # 文本切分器对象

    def upload_by_str(self, data, filename):
        """将传入的字符串进行向量化，并存储到向量库中"""
        # 先得到传入的字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "❌内容已存在，无需重复加载" # 如果md5值已经处理过，则直接返回

        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunks: list[str] = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "Beamus Wayne",
        }

        self.chroma.add_texts( # 内容就加载到向量库中了
            knowledge_chunks, 
            metadatas=[metadata for _ in knowledge_chunks]
        )

        # 
        save_md5(md5_hex)
        return "成功✅内容已加载到向量库中"


if __name__ == "__main__":
    service = KnowledgeBaseService()
    r = service.upload_by_str("你好，我是Beamus Wayne", "test.txt")
    print(r)

    