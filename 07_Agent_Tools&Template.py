from langchain_ollama import ChatOllama
# 关键修改：调整 Agent 相关组件的导入路径
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor as LangChainAgentExecutor
from langchain import hub
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.chains.llm_math.base import LLMMathChain

# --- 初始化部分 ---
# 初始化 Ollama 大模型
llm = ChatOllama(model="qwen3:4b", temperature=0)

# 初始化搜索工具（需要配置 SerpAPI Key）
# 注意：使用 SerpAPI 需要先在 https://serpapi.com/ 获取 API Key，并设置环境变量
search = SerpAPIWrapper()
# 初始化数学计算工具
llm_math_chain = LLMMathChain.from_llm(llm=llm)

# 定义工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="适用于查询当前事件、最新信息、名人近况等需要联网获取的内容"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="适用于解决数学计算问题，比如加减乘除、复杂运算等"
    )
]

# 获取 REACT 提示词模板（如果网络问题拉取失败，可手动定义）
try:
    prompt = hub.pull("hwchase17/react")
except Exception as e:
    print(f"拉取 hub 模板失败: {e}")
    # 手动定义简化版 REACT 提示词模板（备用方案）
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")

# 构建 REACT Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# 构建 Agent 执行器
agent_executor = LangChainAgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    # 新增：设置最大迭代次数，避免无限循环
    max_iterations=10
)

# 执行查询：查询小李子的现任女友
result = agent_executor.invoke({"input": "Who is Leo DiCaprio's girlfriend?"})
print("\n最终回答：", result["output"])