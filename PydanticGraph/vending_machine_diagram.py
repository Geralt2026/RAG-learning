from vending_machine import InsertCoin, vending_machine_graph

diagram = vending_machine_graph.mermaid_code(start_node=InsertCoin)

with open("vending_machine_diagram.md", "w") as f:
    f.write(diagram)