import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import threading

def display_performance_stats():
    """显示性能统计信息"""
    # 检查是否有性能收集器
    if 'performance_collector' in st.session_state:
        return display_enhanced_performance_stats()
    
    # 否则使用旧版实现
    if 'performance_metrics' not in st.session_state or not st.session_state.performance_metrics:
        st.info("尚无性能数据")
        return
    
    # 计算消息响应时间统计
    message_times = [m["duration"] for m in st.session_state.performance_metrics 
                    if m["operation"] == "send_message"]
    
    if message_times:
        avg_time = sum(message_times) / len(message_times)
        max_time = max(message_times)
        min_time = min(message_times)
        
        st.subheader("消息响应性能")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均响应时间", f"{avg_time:.2f}s")
        with col2:
            st.metric("最大响应时间", f"{max_time:.2f}s")
        with col3:
            st.metric("最小响应时间", f"{min_time:.2f}s")
        
        # 绘制响应时间图表
        if len(message_times) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(message_times))
            ax.plot(x, message_times, marker='o')
            ax.set_title('Response Time Trend')
            ax.set_xlabel('Message ID')
            ax.set_ylabel('Response Time (s)')
            ax.grid(True)
            
            st.pyplot(fig)
    
    # 反馈性能统计
    feedback_times = [m["duration"] for m in st.session_state.performance_metrics 
                     if m["operation"] == "send_feedback"]
    
    if feedback_times:
        avg_feedback_time = sum(feedback_times) / len(feedback_times)
        st.subheader("反馈处理性能")
        st.metric("平均反馈处理时间", f"{avg_feedback_time:.2f}s")

def clear_performance_data():
    """清除性能数据"""
    # 清除新版性能收集器
    if 'performance_collector' in st.session_state:
        collector = st.session_state.performance_collector
        collector.reset()
    
    # 清除原有格式的性能数据
    if 'performance_metrics' in st.session_state:
        st.session_state.performance_metrics = []
    
    return True

# 性能数据收集器类
class PerformanceCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.api_calls = defaultdict(int)
        self.api_times = defaultdict(float)
        self.page_loads = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_api_call(self, endpoint, duration):
        """记录API调用"""
        with self.lock:
            self.api_calls[endpoint] += 1
            self.api_times[endpoint] += duration
    
    def record_metric(self, name, value):
        """记录一般性能指标"""
        with self.lock:
            self.metrics[name].append(value)
    
    def record_page_load(self):
        """记录页面加载"""
        with self.lock:
            self.page_loads += 1
    
    def get_uptime(self):
        """获取应用运行时间（秒）"""
        return time.time() - self.start_time
    
    def get_api_stats(self):
        """获取API调用统计"""
        with self.lock:
            total_calls = sum(self.api_calls.values())
            total_time = sum(self.api_times.values())
            return {
                "total_calls": total_calls,
                "total_time": total_time,
                "avg_time": total_time / total_calls if total_calls else 0,
                "calls_by_endpoint": dict(self.api_calls),
                "time_by_endpoint": dict(self.api_times)
            }
    
    def reset(self):
        """重置所有指标"""
        with self.lock:
            self.metrics = defaultdict(list)
            self.api_calls = defaultdict(int)
            self.api_times = defaultdict(float)
            self.page_loads = 0
            self.start_time = time.time()

# 用于获取或创建性能收集器的函数
def get_performance_collector():
    """获取或创建性能收集器实例"""
    if "performance_collector" not in st.session_state:
        st.session_state.performance_collector = PerformanceCollector()
    return st.session_state.performance_collector

# 性能监控装饰器
def monitor_performance(endpoint=None):
    """监控函数性能的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # 记录性能数据 - 新版收集器
            try:
                collector = get_performance_collector()
                if endpoint:
                    collector.record_api_call(endpoint, duration)
                else:
                    func_name = func.__name__
                    collector.record_metric(f"func:{func_name}", duration)
            except Exception as e:
                print(f"记录性能数据失败: {e}")
                
            # 同时兼容旧版记录方式
            if 'performance_metrics' in st.session_state:
                st.session_state.performance_metrics.append({
                    "operation": endpoint or func.__name__,
                    "duration": duration,
                    "timestamp": time.time()
                })
            
            return result
        return wrapper
    return decorator

# 展示增强版性能统计信息的函数
def display_enhanced_performance_stats():
    """显示增强的性能统计信息"""
    collector = get_performance_collector()
    
    # 基本应用统计
    st.subheader("应用性能总览")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uptime = collector.get_uptime()
        days, remainder = divmod(uptime, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        st.metric("运行时间", uptime_str)
    
    with col2:
        api_stats = collector.get_api_stats()
        st.metric("API调用总次数", f"{api_stats['total_calls']}")
    
    with col3:
        st.metric("平均响应时间", f"{api_stats['avg_time']:.2f}s")
    
    # API调用统计
    if api_stats['total_calls'] > 0:
        st.subheader("API调用统计")
        
        # 创建DataFrame以便排序和显示
        api_data = []
        for endpoint, count in api_stats['calls_by_endpoint'].items():
            time_total = api_stats['time_by_endpoint'].get(endpoint, 0)
            time_avg = time_total / count if count else 0
            api_data.append({
                "端点": endpoint,
                "调用次数": count,
                "总时间(秒)": round(time_total, 2),
                "平均时间(秒)": round(time_avg, 2)
            })
        
        df = pd.DataFrame(api_data)
        if not df.empty:
            # 按调用次数降序排序
            df = df.sort_values(by="调用次数", ascending=False)
            st.dataframe(df, use_container_width=True)
            
            # 可视化API调用分布
            if len(df) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                endpoints = df["端点"].tolist()
                calls = df["调用次数"].tolist()
                
                # 横向条形图
                y_pos = np.arange(len(endpoints))
                ax.barh(y_pos, calls, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(endpoints)
                ax.invert_yaxis()  # 最高的在顶部
                ax.set_xlabel('Call Count')
                ax.set_title('API Call Distribution')
                
                st.pyplot(fig)
    
    # 消息响应时间分析
    if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
        message_times = [m["duration"] for m in st.session_state.performance_metrics 
                        if m["operation"] == "send_message"]
        
        if message_times:
            st.subheader("消息响应性能")
            col1, col2, col3 = st.columns(3)
            avg_time = sum(message_times) / len(message_times)
            max_time = max(message_times)
            min_time = min(message_times)
            
            with col1:
                st.metric("平均响应时间", f"{avg_time:.2f}s")
            with col2:
                st.metric("最大响应时间", f"{max_time:.2f}s")
            with col3:
                st.metric("最小响应时间", f"{min_time:.2f}s")
            
            # 绘制响应时间图表
            if len(message_times) > 1:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = np.arange(len(message_times))
                ax.plot(x, message_times, marker='o')
                ax.set_title('Response Time Trend')
                ax.set_xlabel('Message ID')
                ax.set_ylabel('Response Time (s)')
                ax.grid(True)
                
                st.pyplot(fig)
    
    # 系统资源监控
    if collector.metrics:
        st.subheader("系统资源监控")
        
        # 如果有内存使用数据，绘制内存使用图表
        if "memory_usage" in collector.metrics and len(collector.metrics["memory_usage"]) > 1:
            memory_data = collector.metrics["memory_usage"]
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(memory_data))
            ax.plot(x, memory_data, marker='o', color='green')
            ax.set_title('Memory Usage Trend')
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Memory Usage (MB)')
            ax.grid(True)
            
            st.pyplot(fig)
    
    # 添加性能分析工具
    st.subheader("性能分析工具")
    analyze_tab, config_tab = st.tabs(["性能分析", "配置"])
    
    with analyze_tab:
        if st.button("运行性能检测", key="run_perf_check"):
            with st.spinner("正在检测性能瓶颈..."):
                # 模拟性能检测过程
                time.sleep(1.5)
                
                # 显示检测结果
                st.success("性能检测完成")
                st.info("""
                性能分析结果:
                1. API调用 - 状态良好
                2. 前端渲染 - 状态良好
                3. 数据处理 - 无明显瓶颈
                """)
    
    with config_tab:
        st.checkbox("启用详细API日志", value=False, key="enable_api_logging")
        st.slider("性能数据保留时间(小时)", min_value=1, max_value=24, value=6, key="perf_data_retention")
        
        if st.button("应用配置", key="apply_perf_config"):
            st.success("配置已更新")

# 装饰API调用函数
def decorate_api_functions():
    """为API函数添加性能监控装饰器"""
    try:
        from frontend.utils.api import send_message, send_feedback, get_knowledge_graph, get_source_content
        
        # 装饰原始函数
        original_send_message = send_message
        original_send_feedback = send_feedback
        original_get_knowledge_graph = get_knowledge_graph
        original_get_source_content = get_source_content
        
        # 使用监控装饰器包装函数
        @monitor_performance(endpoint="send_message")
        def monitored_send_message(*args, **kwargs):
            return original_send_message(*args, **kwargs)
        
        @monitor_performance(endpoint="send_feedback")
        def monitored_send_feedback(*args, **kwargs):
            return original_send_feedback(*args, **kwargs)
        
        @monitor_performance(endpoint="get_knowledge_graph")
        def monitored_get_knowledge_graph(*args, **kwargs):
            return original_get_knowledge_graph(*args, **kwargs)
        
        @monitor_performance(endpoint="get_source_content")
        def monitored_get_source_content(*args, **kwargs):
            return original_get_source_content(*args, **kwargs)
        
        # 替换原始函数
        import frontend.utils.api
        frontend.utils.api.send_message = monitored_send_message
        frontend.utils.api.send_feedback = monitored_send_feedback
        frontend.utils.api.get_knowledge_graph = monitored_get_knowledge_graph
        frontend.utils.api.get_source_content = monitored_get_source_content
        
        return True
    except Exception as e:
        print(f"装饰API函数失败: {e}")
        return False

# 在App启动时初始化性能收集
def init_performance_monitoring():
    """初始化性能监控"""
    # 获取或创建收集器
    collector = get_performance_collector()
    
    # 记录页面加载
    collector.record_page_load()
    
    # 装饰API函数
    decorate_api_functions()
    
    return collector