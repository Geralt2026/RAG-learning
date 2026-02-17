from fastapi import APIRouter
from models.schemas import FeedbackRequest, FeedbackResponse
from services.chat_service import process_feedback
from utils.performance import measure_performance

# 创建路由器
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
@measure_performance("feedback")
async def feedback(request: FeedbackRequest):
    """
    处理用户对回答的反馈
    
    Args:
        request: 反馈请求
        
    Returns:
        FeedbackResponse: 反馈响应
    """
    result = await process_feedback(
        message_id=request.message_id,
        query=request.query,
        is_positive=request.is_positive,
        thread_id=request.thread_id,
        agent_type=request.agent_type
    )
    
    return FeedbackResponse(**result)