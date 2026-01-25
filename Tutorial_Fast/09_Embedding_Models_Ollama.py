from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="qwen3-embedding:4b")

print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(["我喜欢你", "我爱你", "我想你"]))