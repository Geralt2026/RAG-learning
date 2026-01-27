from dataclasses import dataclass, field
import os
import datasets
import duckdb
import pandas as pd
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, ModelRetry, RunContext


# 创建OpenAI客户端
client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 创建模型
model = OpenAIChatModel("qwen-max", provider=OpenAIProvider(openai_client=client))

@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store(self, value: pd.DataFrame) -> str:
        """将输出存储在deps中并返回引用，例如Out[1]，用于LLM使用。"""
        ref = f'Out[{len(self.output) + 1}]'
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(
                f'错误: {ref} 不是一个有效的变量引用。请检查之前的消息并重试。'
            )
        return self.output[ref]


analyst_agent = Agent[AnalystAgentDeps](
    model,
    deps_type=AnalystAgentDeps,
    instructions='你是一个数据分析师，你的任务是根据用户请求分析数据。',
)


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """从huggingface加载`dataset_name`的`split`数据集。

    Args:
        ctx: Pydantic AI agent RunContext
        path: 数据集名称，格式为`<user_name>/<dataset_name>`
        split: 加载数据集的分割 (默认: "train")
    """
    # 从hf加载数据
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # 避免 DuckDB 把 label 当整数，导致后续 result.df() 报 ConversionException
    if 'label' in dataframe.columns:
        dataframe['label'] = dataframe['label'].astype(str)
    # 从hf加载数据结束

    # 将dataframe存储在deps中并获取引用，例如"Out[1]"
    ref = ctx.deps.store(dataframe)
    # construct a summary of the loaded dataset
    output = [
        f'Loaded the dataset as `{ref}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """在DataFrame上运行DuckDB SQL查询。

    注意，DuckDB SQL中使用的虚拟表名必须是`dataset`。

    Args:
        ctx: Pydantic AI agent RunContext
        dataset: DataFrame的引用字符串
        sql: 使用DuckDB执行的查询
    """
    data = ctx.deps.get(dataset)
    con = duckdb.connect()
    try:
        con.register('dataset', data)
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out_df = pd.DataFrame(rows, columns=columns)
    finally:
        con.unregister('dataset')
        con.close()
    ref = ctx.deps.store(out_df)
    return f'Executed SQL, result is `{ref}`'


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """显示DataFrame的最多5行。"""
    dataset = ctx.deps.get(name)
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == '__main__':
    deps = AnalystAgentDeps()
    result = analyst_agent.run_sync(
        user_prompt='统计数据集`cornell-movie-review-data/rotten_tomatoes`中多少条负面评论',
        deps=deps,
    )
    print(result.output)