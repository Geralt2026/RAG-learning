"""分镜脚本数据模型。"""
from pydantic import BaseModel, Field


class ShotConstraints(BaseModel):
    """单个镜头的约束条件。"""
    tags: list[str] = Field(default_factory=list)
    duration_range: tuple[float, float] = (3.0, 10.0)  # (min_sec, max_sec)
    shot_type: str | None = None  # wide | medium | close | detail
    resolution: str | None = None


class ShotItem(BaseModel):
    """分镜中的单个镜头。"""
    slot: int
    type: str  # "library" | "user_upload"
    user_slot_id: str | None = None  # 当 type=user_upload 时对应前端上传的 key
    constraints: ShotConstraints = Field(default_factory=ShotConstraints)


class ShotList(BaseModel):
    """完整分镜脚本。"""
    name: str = ""
    version: str = "1.0"
    total_duration_target: float = 60.0  # 秒
    shots: list[ShotItem] = Field(default_factory=list)
