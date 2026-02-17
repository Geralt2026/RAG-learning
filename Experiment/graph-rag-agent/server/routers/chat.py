from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
from models.schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse
from services.chat_service import process_chat, process_chat_stream
from services.agent_service import agent_manager, format_execution_log
from utils.performance import measure_performance

# 创建路由器
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@measure_performance("chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求
    
    Args:
        request: 聊天请求
        
    Returns:
        ChatResponse: 聊天响应
    """
    result = await process_chat(
        message=request.message,
        session_id=request.session_id,
        debug=request.debug,
        agent_type=request.agent_type,
        use_deeper_tool=request.use_deeper_tool,
        show_thinking=request.show_thinking
    )
    
    if request.debug and "execution_log" in result:
        # 格式化执行日志
        result["execution_log"] = format_execution_log(result["execution_log"])
    
    return ChatResponse(**result)

def serialize_log_entry(log_entry):
    """将日志条目转换为可序列化的格式"""
    if isinstance(log_entry, dict):
        result = {}
        for key, value in log_entry.items():
            # 处理输入字段
            if key == "input":
                if hasattr(value, "content"):
                    # 处理 Message 对象
                    result[key] = {"content": value.content}
                elif isinstance(value, dict):
                    # 处理字典，但需要确保内部没有不可序列化的对象
                    result[key] = {}
                    for k, v in value.items():
                        if hasattr(v, "content"):
                            result[key][k] = {"content": v.content}
                        else:
                            try:
                                # 验证可序列化性 - 使用导入的 json 模块
                                json.dumps(v)
                                result[key][k] = v
                            except:
                                result[key][k] = str(v)
                else:
                    # 其他情况，使用字符串表示
                    result[key] = str(value)
            # 处理输出字段
            elif key == "output":
                if hasattr(value, "content"):
                    # 处理 Message 对象
                    result[key] = {"content": value.content}
                else:
                    # 使用字符串表示
                    result[key] = str(value)
            # 处理其他字段
            else:
                result[key] = value
        return result
    return str(log_entry)

@router.post("/chat/stream")
async def chat_stream(request: Request):
    """流式响应聊天请求"""
    # 解析请求数据
    data = await request.json()
    message = data.get("message")
    session_id = data.get("session_id")
    debug = data.get("debug", False)
    agent_type = data.get("agent_type", "hybrid_agent")
    use_deeper_tool = data.get("use_deeper_tool", True)
    show_thinking = data.get("show_thinking", False)
    
    # 设置流式响应
    async def event_generator():
        try:
            # 确保明确设置格式为SSE，并且使用已导入的 json 模块
            yield "data: " + json.dumps({"status": "start"}) + "\n\n"
            
            # 处理消息流
            execution_log = []
            
            async for chunk in process_chat_stream(
                message=message,
                session_id=session_id,
                debug=debug,
                agent_type=agent_type,
                use_deeper_tool=use_deeper_tool,
                show_thinking=show_thinking
            ):
                # 检查是否是字典格式
                if isinstance(chunk, dict):
                    # 提取执行轨迹（如果有）
                    if "execution_log" in chunk and debug:
                        log_entry = chunk["execution_log"]
                        execution_log.append(log_entry)
                        # 序列化日志条目，避免非JSON可序列化对象
                        serialized_log = serialize_log_entry(log_entry)
                        try:
                            yield "data: " + json.dumps({
                                "status": "execution_log",
                                "content": serialized_log
                            }) + "\n\n"
                        except Exception as json_error:
                            print(f"执行日志序列化错误: {json_error}")
                            # 尝试一个更简单的方法
                            yield "data: " + json.dumps({
                                "status": "execution_log",
                                "content": {"simplified": str(log_entry)}
                            }) + "\n\n"
                    # 继续正常流程
                    elif "status" in chunk:
                        try:
                            yield "data: " + json.dumps(chunk) + "\n\n"
                        except Exception as json_error:
                            print(f"状态序列化错误: {json_error}")
                            yield "data: " + json.dumps({
                                "status": "error", 
                                "message": "状态序列化错误"
                            }) + "\n\n"
                    else:
                        # 转换为文本块
                        try:
                            yield "data: " + json.dumps({
                                "status": "token", 
                                "content": str(chunk)
                            }) + "\n\n"
                        except Exception as json_error:
                            print(f"令牌序列化错误: {json_error}")
                else:
                    # 普通文本块
                    try:
                        yield "data: " + json.dumps({
                            "status": "token", 
                            "content": chunk
                        }) + "\n\n"
                    except Exception as json_error:
                        print(f"普通文本序列化错误: {json_error}")
                
            # 最后发送完整的执行日志
            if debug and execution_log:
                try:
                    # 序列化执行日志
                    serialized_logs = [serialize_log_entry(log) for log in execution_log]
                    yield "data: " + json.dumps({
                        "status": "execution_logs",
                        "content": serialized_logs
                    }) + "\n\n"
                except Exception as json_error:
                    print(f"执行日志组序列化错误: {json_error}")
                    yield "data: " + json.dumps({
                        "status": "execution_logs",
                        "content": [{"simplified": "日志序列化失败"}]
                    }) + "\n\n"
                
            # 发送完成事件
            yield "data: " + json.dumps({"status": "done"}) + "\n\n"
        except Exception as e:
            # 发送错误事件
            print(f"事件生成器错误: {e}")
            yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"
    
    # 返回流式响应
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 阻止Nginx缓冲
        }
    )

@router.post("/clear", response_model=ClearResponse)
async def clear_chat(request: ClearRequest):
    """
    清除聊天历史
    
    Args:
        request: 清除请求
        
    Returns:
        ClearResponse: 清除响应
    """
    result = agent_manager.clear_history(request.session_id)
    return ClearResponse(**result)