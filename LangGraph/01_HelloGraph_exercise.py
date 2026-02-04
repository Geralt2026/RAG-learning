from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState(TypedDict):
    message : str

def gretting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting to the message"""

    state['message'] = state["message"] + ", you're doing an amazing job learning LangGraph!"

    return state

graph = StateGraph(AgentState) # 创建一个LangGraph的状态图，并指定其类型为AgentState

# LangGraph 里的图结构：节点是处理函数，边是流转关系，整张图在运行时会按边在节点之间传递「状态」。

graph.add_node("greeter", gretting_node) # 添加一个节点，名称为"greeter"，节点函数为gretting_node

# 添加边
graph.add_edge(START, "greeter") # 从起点到greeter节点
graph.add_edge("greeter", END) # 从greeter节点到终点

app = graph.compile() # 编译图

result = app.invoke({"message": "Bob"}) # 执行图，输入状态为{"message": "Bob"}
print(result) # 打印结果