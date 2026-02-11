# agent: 编排层
# - tools.py: 对 services 的封装（解析、选片、随机、查重、拼接、注册指纹、BGM）
# - graph.py: 纯 Python 合成流程入口 run_synthesis；可选 use_langgraph=True 走 LangGraph
# - graph_langgraph.py: LangGraph 版流程 run_synthesis_langgraph（需 pip install langgraph）
