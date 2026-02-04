from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    message : str

def gretting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting to the message"""

    state['message'] = "Hey " + state["message"] + ", how is your day going?"

    return state

graph = StateGraph(AgentState) # 参数解释：状态类型

graph.add_node("greeter", gretting_node) # 参数解释：名字、节点函数

graph.add_edge(START, "greeter") # 参数解释：起点、终点
graph.add_edge("greeter", END) # 参数解释：起点、终点

app = graph.compile() # 编译图

result = app.invoke({"message": "John"}) # 输入状态
print(result)
print("--------------------------------")
print(result["message"])