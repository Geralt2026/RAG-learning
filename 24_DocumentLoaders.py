# 文档加载器提供了一套标准接口，用于将不同来源（如CSV、PDF 或JSON等）的数据读取为LangChain 的文档格式。
# 这确保了无论数据来源如何，都能对其进行一致性处理。
# 文档加载器（内置或自行实现）需实现BaseLoader接口。

# Class Document，是LangChain内文档的统一载体，所有文档加载器最终返回此类的实例。
# 一个基础的Document类实例，基于如下代码创建：

from langchain_core.documents import Document

document = Document(
    page_content="Hello World", metadata={"source": "http://example.com"}
    )


# 可以看到,Document类其核心记录了: page_content:文档内容   metadata: 文档元数据(字典)

# 不同的文档加载器可能定义了不同的参数，但是其都实现了统一的接口（方法）。
# load()：一次性加载全部文档
# lazy_load()：延迟流式传输文档，对大型数据集很有用，避免内存溢出。
