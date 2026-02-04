import math
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class AgentState(TypedDict, total=False):
    value: List[int]
    name: str
    result: str
    operation: str

def process_values(state: AgentState) -> AgentState:
    """this function handles multiple different inputs"""

    op = state.get('operation')
    if op == '+':
        state['result'] = f"Hi there {state['name']}! Your sum = {sum(state['value'])}"
    elif op == '*':
        state['result'] = f"Hi there {state['name']}! Your product = {math.prod(state['value'])}"
    else:
        state['result'] = "Invalid operation"

    return state

graph = StateGraph(AgentState)

graph.add_node("processor", process_values)
graph.set_entry_point("processor")
graph.set_finish_point("processor")

app = graph.compile()

result = app.invoke({"value": [1, 2, 3], "name": "John Wick", "operation": "+"})
print(result)

result = app.invoke({"value": [1, 2, 3], "name": "John Wick", "operation": "*"})
print(result)

result = app.invoke({"value": [1, 2, 3], "name": "John Wick"})
print(result)