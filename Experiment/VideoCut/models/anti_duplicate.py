"""防重复相关数据模型。"""
from uuid import UUID
from pydantic import BaseModel, Field


class RandomParams(BaseModel):
    """单次合成使用的随机化参数（防重复）。"""
    seed: int = 0
    transition_type: str = "fade"
    transition_duration: float = 0.5
    scale: float = 1.0
    crop_ratio: float = 1.0
    filter_preset: str = "none"  # none | slight_contrast | slight_warm
    bgm_index: int = 0


class CheckResult(BaseModel):
    """成片查重结果。"""
    passed: bool = True
    similar_video_ids: list[UUID] = Field(default_factory=list)
    max_similarity: float = 0.0
    message: str = ""


class VideoFingerprint(BaseModel):
    """成片指纹（帧哈希序列等），用于相似度比对。"""
    video_id: UUID
    frame_hashes: list[str] = Field(default_factory=list)  # 按时间序的 pHash/dHash
    duration_sec: float = 0.0
