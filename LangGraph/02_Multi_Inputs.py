from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict

class AgentState(TypedDict):
    value: List[int]
    name: str
    result: str

def process_values(state: AgentState) -> AgentState:
    """this function handles multiple different inputs"""
    print(state)

    state['result'] = f"Hi there {state['name']}! Your sum = {sum(state['value'])}"
    
    print(state)
    return state

graph = StateGraph(AgentState)

graph.add_node("processor", process_values)
graph.set_entry_point("processor") # Set the starting node
graph.set_finish_point("processor") # Set the ending node

app = graph.compile() # Compiling the graph 

mermaid_syntax = app.get_graph().draw_mermaid()
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
draw_mermaid_png(mermaid_syntax, output_file_path="LangGraph/graph.png")

answers = app.invoke({"value": [1, 2, 3], "name": "John Wick"})
print("---------------answers---------------")
print(answers)
print("---------answers['result']-----------")
print(answers["result"])

