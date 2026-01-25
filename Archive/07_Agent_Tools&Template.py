from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
from langchain_community.chains.llm_math.base import LLMMathChain
# 导入Tavily搜索工具（替代SerpAPI）
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# --- 初始化核心组件 ---
# 初始化Ollama大模型
llm = ChatOllama(model="qwen3:4b", temperature=0)

# 1. 初始化Tavily搜索工具（替换SerpAPI）
# 方式1：通过环境变量设置API Key（推荐）
# Windows: set TAVILY_API_KEY=你的Tavily API Key
# Mac/Linux: export TAVILY_API_KEY=你的Tavily API Key
# 方式2：直接在代码中设置（测试用）
import os
os.environ["TAVILY_API_KEY"] = "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"  # 替换为你的Key

tavily_search = TavilySearchAPIWrapper()

# 2. 初始化数学计算工具（无需API Key）
llm_math_chain = LLMMathChain.from_llm(llm=llm)

# --- 定义工具列表（无SerpAPI依赖） ---
tools = [
    Tool(
        name="Search",
        func=tavily_search.run,
        description="适用于查询当前事件、名人近况、最新资讯等需要联网获取的信息"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="适用于解决数学计算问题，比如加减乘除、复杂运算等"
    )
]

# --- 构建Agent并运行 ---
# 拉取REACT提示词模板（失败则用本地备用模板）
try:
    prompt = hub.pull("hwchase17/react")
except Exception as e:
    print(f"拉取hub模板失败，使用本地备用模板: {e}")
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

# 构建Agent和执行器
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10  # 避免无限循环
)

# 测试运行（查询小李子女友，无需SerpAPI）
result = agent_executor.invoke({"input": "Who is Leo DiCaprio's girlfriend in 2026?"})
print("\n最终回答：", result["output"])