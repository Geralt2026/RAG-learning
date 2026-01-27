from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_core.documents import Document
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from vector_stores import VectorStoreService
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history


def print_prompt(prompt):
    print("=" * 20, prompt.to_string(), "=" * 20)
    return prompt


class RAGService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            DashScopeEmbeddings(model=config.embedding_model)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "以我提供的已知参考为主，简洁和专业的回答用户问题。参考资料{context}。",
                ),
                ("system", "并且我提供用户的对话历史记录如下："),
                MessagesPlaceholder(variable_name="history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model)
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_docs(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_docs = ""
            for doc in docs:
                formatted_docs += (
                    f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
                )

            formatted_docs += "--------------------------------\n"
            return formatted_docs  # 返回格式化后的参考资料

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever)
                | retriever
                | format_docs,
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == "__main__":
    session_config = config.session_config

    result = RAGService().chain.invoke(
        {"input": "我150斤，适合穿什么？"}, config=session_config
    )
    print(result)
