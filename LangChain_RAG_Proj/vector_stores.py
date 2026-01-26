from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        '''
        嵌入模型的传入
        '''
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=self.embedding,
            persist_directory=config.chroma_persist_directory,
        )
    
    def get_retriever(self):
        '''获取检索器, 方便加入chain'''
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
    
if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever =  VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    print(retriever.invoke("我体重180斤，尺码推荐"))

