from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class AgentState(TypedDict, total=False):
    name: str
    age: int
    final: str

def first_node(state: AgentState) -> AgentState:
    """This is the first node of our sequence"""

    state['final'] = f"Hi {state['name']}"
    return state

def second_node(state: AgentState) -> AgentState:
    """This is the second node of our sequence"""

    state['final'] = state['final'] + f" You are {state['age']} years old!"
    return state


graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)

graph.set_entry_point("first_node")
graph.add_edge("first_node", "second_node")
graph.set_finish_point("second_node")

app = graph.compile()

mermaid_syntax = app.get_graph().draw_mermaid()
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
draw_mermaid_png(mermaid_syntax, output_file_path="LangGraph/graph.png")

answers = app.invoke({"name": "John", "age": 20})
print(answers)
print(answers["final"])

