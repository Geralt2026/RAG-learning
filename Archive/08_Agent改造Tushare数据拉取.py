import tushare as ts
import pandas as pd
from sqlalchemy import create_engine, text
import dashscope # 阿里百炼 SDK，如使用豆包请替换为 volcenginesdkarkruntime
import schedule
import time
from datetime import datetime

# --- 基础配置 ---
TUSHARE_TOKEN = '9f4ccc450304a3942be9e869a00d602b365f033435d9b9c517134e23'
DB_URL = "mysql+pymysql://root:1234@localhost:3306/db_data?charset=utf8mb4"
ALIBABA_API_KEY = "sk-22daeee007d74be98634b70c63396c5e"

# 初始化
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
engine = create_engine(DB_URL, pool_recycle=60, pool_size=10)
dashscope.api_key = ALIBABA_API_KEY

# 全局变量控制频率
current_interval = 10 

def ask_llm(prompt):
    """调用大模型获取决策或分析结果"""
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo', # 或 'doubao-pro-4k' 等
            prompt=prompt
        )
        if response.status_code == 200:
            return response.output.text
        return "ERROR"
    except Exception as e:
        return f"LLM_FAILED: {str(e)}"

def save_to_mysql(df, table_name):
    """将数据存入对应的 MySQL 表"""
    try:
        # 使用 append 模式，如果表不存在会自动创建
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"[{datetime.now()}] 成功存入表: {table_name}, 条数: {len(df)}")
    except Exception as e:
        print(f"数据库写入失败: {e}")
        # 触发异常监控决策
        decision = ask_llm(f"数据库写入报错: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def save_full_to_mysql(df, table_name):
    """全量覆盖保存：表不存在自动创建，存在则替换为最新全量数据"""
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"[{datetime.now()}] 全量覆盖表: {table_name}, 条数: {len(df)}")
    except Exception as e:
        print(f"数据库全量写入失败: {e}")
        decision = ask_llm(f"数据库全量写入报错: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def download_stock_basic_all_fields():
    """按官方接口，全量下载股票基本信息的所有字段，保存到 db_data.stock_basic"""
    try:
        limit = 5000
        offset = 0
        frames = []
        while True:
            df = pro.stock_basic(limit=limit, offset=offset)
            if df is None or df.empty:
                break
            frames.append(df)
            print(f"[{datetime.now()}] 拉取 stock_basic 批次 offset={offset}, size={len(df)}")
            offset += limit
            time.sleep(1)
        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            save_full_to_mysql(full_df, "stock_basic")
        else:
            print("stock_basic 未获取到数据")
    except Exception as e:
        print(f"下载 stock_basic 失败: {e}")
        decision = ask_llm(f"Tushare stock_basic 全量下载异常: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def download_namechange_all_fields():
    """全量下载股票名称变更历史，保存到 db_data.namechange"""
    try:
        limit = 5000
        offset = 0
        frames = []
        while True:
            df = pro.namechange(limit=limit, offset=offset)
            if df is None or df.empty:
                break
            frames.append(df)
            print(f"[{datetime.now()}] 拉取 namechange 批次 offset={offset}, size={len(df)}")
            offset += limit
            time.sleep(1)
        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            save_full_to_mysql(full_df, "namechange")
        else:
            print("namechange 未获取到数据")
    except Exception as e:
        print(f"下载 namechange 失败: {e}")
        decision = ask_llm(f"Tushare namechange 全量下载异常: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def download_stock_company_all_fields():
    """全量下载上市公司基本信息，保存到 db_data.stock_company"""
    try:
        limit = 5000
        offset = 0
        frames = []
        while True:
            # 部分账户需指定 exchange，尝试不指定获取全量
            df = pro.stock_company(limit=limit, offset=offset)
            if df is None or df.empty:
                break
            frames.append(df)
            print(f"[{datetime.now()}] 拉取 stock_company 批次 offset={offset}, size={len(df)}")
            offset += limit
            time.sleep(1)
        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            save_full_to_mysql(full_df, "stock_company")
        else:
            print("stock_company 未获取到数据")
    except Exception as e:
        print(f"下载 stock_company 失败: {e}")
        decision = ask_llm(f"Tushare stock_company 全量下载异常: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def download_new_share_all_fields():
    """全量下载新股上市数据，保存到 db_data.new_share"""
    try:
        df = pro.new_share()
        if df is None or df.empty:
            print("new_share 未获取到数据")
            return
        save_full_to_mysql(df, "new_share")
    except Exception as e:
        print(f"下载 new_share 失败: {e}")
        decision = ask_llm(f"Tushare new_share 全量下载异常: {str(e)}。请给出简短处理建议。")
        print(f"Agent 建议: {decision}")

def get_data_and_analyze():
    global current_interval
    now = datetime.now()
    
    # 仅在交易时段运行 (周一至周五 09:00-15:00)
    if now.weekday() >= 5 or not (9 <= now.hour < 15):
        return

    today_str = now.strftime('%Y%m%d')
    print(f"\n--- 开始执行任务: {now.strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        # 1. 爬取 daily_basic (每日指标)
        df_daily = pro.daily_basic(trade_date=today_str)
        if not df_daily.empty:
            save_to_mysql(df_daily, 'daily_basic')
            
            # 智能分析：涨幅快讯
            # 注意：daily_basic可能不含pct_chg，建议结合daily接口或计算
            summary = ask_llm(f"这是今日部分股票数据摘要: {df_daily.head(10).to_string()}。请生成一句20字盘中行情快讯。")
            print(f"【盘中快讯】: {summary}")

            # 动态调频：根据平均换手率或波动判断
            avg_turnover = df_daily['turnover_rate'].mean()
            decision_freq = ask_llm(f"当前市场平均换手率为 {avg_turnover}%。如果市场极度活跃，请回复2，否则回复10。只需返回数字。")
            try:
                new_interval = int(''.join(filter(str.isdigit, decision_freq)))
                if new_interval != current_interval:
                    current_interval = new_interval
                    print(f"!!! 频率已动态调整为: {current_interval} 分钟")
                    # 更新定时器
                    schedule.clear()
                    schedule.every(current_interval).minutes.do(get_data_and_analyze)
            except:
                pass

        # 2. 爬取 stk_limit (涨跌停板数据)
        df_limit = pro.stk_limit(trade_date=today_str)
        if not df_limit.empty:
            save_to_mysql(df_limit, 'stk_limit')

    except Exception as e:
        # 异常监控决策
        error_msg = str(e)
        decision = ask_llm(f"Tushare接口调用异常: {error_msg}。判断是该'重试'还是'等待'？只回复两个字。")
        print(f"Agent 异常决策: {decision}")
        if "重试" in decision:
            time.sleep(30)
            return get_data_and_analyze()

# --- 定时器设置 ---
schedule.every(current_interval).minutes.do(get_data_and_analyze)

print(f"股票 Agent 已启动。当前监控频率: {current_interval} 分钟/次")
# 启动时先执行一次
download_stock_basic_all_fields()
download_namechange_all_fields()
download_stock_company_all_fields()
download_new_share_all_fields()
get_data_and_analyze()

while True:
    schedule.run_pending()
    time.sleep(1)
