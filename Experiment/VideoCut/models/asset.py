"""素材库数据模型。"""
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class AssetType(str, Enum):
    UPLOAD = "upload"  # 我方上传
    SCENIC = "scenic"   # 景区接入


class ShotType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE = "close"
    DETAIL = "detail"


class AssetMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list, description="场景、地点等标签")
    shot_type: ShotType | None = None
    license: str = "internal"  # internal | cc | purchased


class Asset(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: AssetType
    source_file_path: str
    duration_sec: float
    resolution: str = "1080p"
    metadata: AssetMetadata = Field(default_factory=AssetMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class Clip(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID
    in_point_sec: float
    out_point_sec: float
    duration_sec: float
    variant_ids: list[UUID] = Field(default_factory=list)  # 可选：关联的 ClipVariant
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class TransformType(str, Enum):
    ORIGINAL = "original"
    MIRROR_H = "mirror_h"
    MIRROR_V = "mirror_v"
    SPEED_09 = "speed_0.9"
    SPEED_105 = "speed_1.05"
    SPEED_11 = "speed_1.1"
    CROP_CENTER_98 = "crop_center_98"
    CROP_CENTER_95 = "crop_center_95"


class ClipVariant(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    clip_id: UUID
    transform: TransformType = TransformType.ORIGINAL
    output_path: str = ""
    duration_sec: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}
