from langgraph.graph import StateGraph, MessagesState, START, END


def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}


graph = StateGraph(MessagesState)

# 添加节点
graph.add_node(mock_llm)

# 添加边
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)

# 编译
graph = graph.compile()

# 执行
result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(type(result))
print(result["messages"])

