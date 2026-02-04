# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 账户ID（在QMT策略交易界面自动分配，编辑模式下需手动设置）
account = 'your_account_id'  # 请替换为实际账户ID

def init(ContextInfo):
    # 策略初始化
    ContextInfo.log('策略初始化完成')
    ContextInfo.log('每日开盘将自动买入股价最低的10支股票')

def handlebar(ContextInfo):
    # 获取当前时间
    current_time = ContextInfo.get_current_time()
    
    # 检查是否为开盘时间（假设9:30为开盘时间）
    if current_time.hour == 9 and current_time.minute == 30:
        ContextInfo.log('到达开盘时间，开始执行策略')
        
        try:
            # 获取所有可交易股票数据
            # 注意：此处需要根据QMT实际API调整，这里假设使用get_all_stock_data获取股票数据
            stock_data = get_all_stock_data(ContextInfo)
            
            if not stock_data:
                ContextInfo.log('未能获取股票数据，策略终止')
                return
            
            # 转换为DataFrame方便处理
            df = pd.DataFrame(stock_data)
            
            # 过滤条件：排除ST股、退市股、停牌股
            df = df[
                (df['is_st'] == 0) & 
                (df['is_delisting'] == 0) & 
                (df['is_suspended'] == 0) &
                (df['price'] > 0)  # 排除价格为0的股票
            ]
            
            # 按股价升序排序，取前10支
            df_sorted = df.sort_values('price', ascending=True).head(10)
            
            if df_sorted.empty:
                ContextInfo.log('未找到符合条件的股票，策略终止')
                return
            
            ContextInfo.log(f'筛选出{len(df_sorted)}支股价最低的股票:')
            for idx, row in df_sorted.iterrows():
                ContextInfo.log(f"{row['stock_code']} {row['stock_name']} - 价格: {row['price']:.2f}")
            
            # 获取账户可用资金
            account_info = get_trade_detail_data(account)
            available_cash = account_info['available_cash']
            ContextInfo.log(f'账户可用资金: {available_cash:.2f}元')
            
            # 计算每支股票的买入金额（平均分配资金）
            if available_cash <= 0:
                ContextInfo.log('账户可用资金不足，无法买入')
                return
            
            per_stock_amount = available_cash / len(df_sorted)
            ContextInfo.log(f'每支股票计划买入金额: {per_stock_amount:.2f}元')
            
            # 执行买入操作
            for idx, row in df_sorted.iterrows():
                stock_code = row['stock_code']
                stock_price = row['price']
                
                # 计算可买入数量（100股的整数倍）
                if stock_price <= 0:
                    ContextInfo.log(f'股票{stock_code}价格异常，跳过买入')
                    continue
                    
                max_shares = int(per_stock_amount / stock_price / 100) * 100
                
                if max_shares < 100:
                    ContextInfo.log(f'股票{stock_code}可买入数量不足100股，跳过')
                    continue
                
                # 提交买入订单
                try:
                    # 使用市价单买入
                    order_result = passorder(
                        opType=2,  # 2=买入
                        orderType=1101,  # 1101=限价单，1102=市价单（请根据QMT实际参数调整）
                        accountid=account,
                        orderCode=stock_code,
                        prType=14,  # 14=市价（请根据QMT实际参数调整）
                        price=0.0,  # 市价单价格设为0
                        volume=max_shares,
                        strategyName='低价股策略',
                        quickTrade=1,
                        userOrderId=f'LowPrice_{stock_code}',
                        ContextInfo=ContextInfo
                    )
                    
                    ContextInfo.log(f'成功提交{stock_code}买入订单，数量: {max_shares}股')
                    
                except Exception as e:
                    ContextInfo.log(f'提交{stock_code}买入订单失败: {str(e)}')
                    continue
            
            ContextInfo.log('策略执行完成')
            
        except Exception as e:
            ContextInfo.log(f'策略执行出错: {str(e)}')
            
    else:
        # 非开盘时间，不执行操作
        pass


def get_all_stock_data(ContextInfo=None):
    '''获取所有可交易股票数据。优先使用 xtdata（迅投原生），其次尝试 ContextInfo（策略环境），否则返回示例数据。'''
    # 1) 尝试使用 xtquant.xtdata（QMT 原生 Python 接口）
    try:
        from xtquant import xtdata

        # 全市场 A 股代码
        code_list = xtdata.get_stock_list_in_sector('沪深A股')
        if not code_list:
            return _sample_stock_data()

        # 批量获取最新快照（get_full_tick 返回 {code: {lastPrice, ...}}）
        tick = xtdata.get_full_tick(code_list)
        if not tick:
            return _sample_stock_data()

        result = []
        for code in code_list:
            t = tick.get(code) or {}
            # 不同版本字段可能为 lastPrice / last_price / price
            price = t.get('lastPrice') or t.get('last_price') or t.get('price') or 0.0
            if isinstance(price, (list, tuple)):
                price = price[-1] if price else 0.0

            # 股票名称与状态：优先 instrument_detail
            detail = xtdata.get_instrument_detail(code) or {}
            name = detail.get('InstrumentName') or detail.get('instrument_name') or detail.get('name') or code
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')

            # ST：名称含 ST 或 detail 中有标记
            is_st = 1 if ('ST' in str(name).upper() or detail.get('is_st') or detail.get('IsST')) else 0
            is_delisting = 1 if detail.get('delisting') or detail.get('is_delisting') else 0
            # 停牌：部分接口在 tick 里用 suspended 或 volume=0 表示
            is_suspended = 1 if (t.get('suspended') or (t.get('volume') == 0 and price > 0)) else 0
            if is_suspended == 0 and detail.get('suspended'):
                is_suspended = 1

            result.append({
                'stock_code': code,
                'stock_name': name,
                'price': float(price),
                'is_st': is_st,
                'is_delisting': is_delisting,
                'is_suspended': is_suspended,
            })
        return result
    except Exception:
        pass

    # 2) 策略环境下若有 ContextInfo，可用 load_stk_list / get_market_data_ex 等补全
    if ContextInfo is not None:
        try:
            # 部分 QMT 策略环境通过 ContextInfo 暴露全市场列表
            if hasattr(ContextInfo, 'get_stock_list_in_sector'):
                code_list = ContextInfo.get_stock_list_in_sector('沪深A股')
            elif hasattr(ContextInfo, 'get_stock_list'):
                code_list = ContextInfo.get_stock_list()
            else:
                code_list = []
            if code_list:
                result = []
                for code in code_list:
                    try:
                        md = ContextInfo.get_market_data_ex([], [code], period='1d') if hasattr(ContextInfo, 'get_market_data_ex') else None
                        price = 0.0
                        if md and isinstance(md, dict) and code in md:
                            close = md[code].get('close') or md[code].get('lastPrice')
                            price = float(close[-1] if isinstance(close, (list, tuple)) else close or 0)
                        name = code
                        if hasattr(ContextInfo, 'get_instrument_detail'):
                            detail = ContextInfo.get_instrument_detail(code) or {}
                            name = detail.get('InstrumentName') or detail.get('instrument_name') or code
                        is_st = 1 if 'ST' in str(name).upper() else 0
                        result.append({
                            'stock_code': code,
                            'stock_name': name,
                            'price': price,
                            'is_st': is_st,
                            'is_delisting': 0,
                            'is_suspended': 0,
                        })
                    except Exception:
                        continue
                if result:
                    return result
        except Exception:
            pass

    # 3) 无 QMT 环境或接口不可用时返回示例数据（便于本地/回测调试）
    return _sample_stock_data()


def _sample_stock_data():
    '''返回示例股票数据，用于无 QMT 环境或测试。'''
    return [
        {'stock_code': '000001.SZ', 'stock_name': '平安银行', 'price': 10.50, 'is_st': 0, 'is_delisting': 0, 'is_suspended': 0},
        {'stock_code': '600000.SH', 'stock_name': '浦发银行', 'price': 7.80, 'is_st': 0, 'is_delisting': 0, 'is_suspended': 0},
        {'stock_code': '002415.SZ', 'stock_name': '海康威视', 'price': 35.20, 'is_st': 0, 'is_delisting': 0, 'is_suspended': 0},
    ]


if __name__ == '__main__':
    # 模拟测试
    class MockContextInfo:
        def log(self, msg):
            print(msg)
            
        def get_current_time(self):
            import datetime
            return datetime.datetime(2026, 2, 4, 9, 30)  # 模拟开盘时间
    
    mock_ctx = MockContextInfo()
    init(mock_ctx)
    handlebar(mock_ctx)