"""素材库服务：检索、选片、返回可用于合成的 Clip/ClipVariant。"""
import random
from pathlib import Path
from uuid import UUID

from config.settings import get_settings
from models.asset import Asset, Clip, ClipVariant, AssetType, TransformType, ShotType
from models.shot_list import ShotConstraints


# 演示用内存存储；生产可替换为 SQLAlchemy + PostgreSQL
_assets: dict[UUID, Asset] = {}
_clips: dict[UUID, Clip] = {}
_variants: dict[UUID, ClipVariant] = {}
_clips_by_asset: dict[UUID, list[UUID]] = {}
_variants_by_clip: dict[UUID, list[UUID]] = {}


def _ensure_dirs() -> None:
    s = get_settings()
    for d in (s.assets_path, s.clips_path, s.storage_path):
        d.mkdir(parents=True, exist_ok=True)


def add_asset(asset: Asset) -> Asset:
    _ensure_dirs()
    _assets[asset.id] = asset
    _clips_by_asset[asset.id] = []
    return asset


def add_clip(clip: Clip) -> Clip:
    _clips[clip.id] = clip
    aid = clip.asset_id
    if aid not in _clips_by_asset:
        _clips_by_asset[aid] = []
    _clips_by_asset[aid].append(clip.id)
    _variants_by_clip[clip.id] = list(clip.variant_ids)
    return clip


def add_clip_variant(variant: ClipVariant) -> ClipVariant:
    _variants[variant.id] = variant
    cid = variant.clip_id
    if cid in _clips:
        if _clips[cid].variant_ids is None:
            _clips[cid].variant_ids = []
        if variant.id not in _clips[cid].variant_ids:
            _clips[cid].variant_ids.append(variant.id)
    if cid not in _variants_by_clip:
        _variants_by_clip[cid] = []
    if variant.id not in _variants_by_clip[cid]:
        _variants_by_clip[cid].append(variant.id)
    return variant


def get_asset(asset_id: UUID) -> Asset | None:
    return _assets.get(asset_id)


def get_clip(clip_id: UUID) -> Clip | None:
    return _clips.get(clip_id)


def get_clip_variant(variant_id: UUID) -> ClipVariant | None:
    return _variants.get(variant_id)


def search_clips(
    tags: list[str] | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
    shot_type: str | None = None,
    asset_type: AssetType | None = None,
    limit: int = 50,
) -> list[Clip]:
    """按约束检索片段；若 tags 为空则按时长与 shot_type 过滤。"""
    candidates: list[Clip] = []
    for cid, clip in _clips.items():
        asset = _assets.get(clip.asset_id)
        if not asset:
            continue
        if asset_type is not None and asset.type != asset_type:
            continue
        if duration_min is not None and clip.duration_sec < duration_min:
            continue
        if duration_max is not None and clip.duration_sec > duration_max:
            continue
        if shot_type and asset.metadata.shot_type and asset.metadata.shot_type.value != shot_type:
            continue
        if tags:
            if not any(t in asset.metadata.tags for t in tags):
                continue
        candidates.append(clip)
    return candidates[:limit]


def pick_clips_for_constraints(
    constraints: ShotConstraints,
    seed: int = 0,
    prefer_variant: bool = True,
) -> list[tuple[Clip, ClipVariant | None]]:
    """
    根据分镜约束选片，并随机选择是否使用某条 Clip 的 Variant。
    返回 [(Clip, ClipVariant|None)]，若该 Clip 无 Variant 则第二项为 None。
    """
    rng = random.Random(seed)
    duration_min, duration_max = constraints.duration_range
    clips = search_clips(
        tags=constraints.tags if constraints.tags else None,
        duration_min=duration_min,
        duration_max=duration_max,
        shot_type=constraints.shot_type,
        limit=100,
    )
    if not clips:
        return []

    # 随机选一条或若干条（这里简化：每个 slot 只取一条，由调用方按 slot 多次调用或取第一个）
    chosen = rng.choice(clips)
    variant = None
    if prefer_variant and chosen.variant_ids:
        vid = rng.choice(chosen.variant_ids)
        variant = _variants.get(vid)
    return [(chosen, variant)]


def get_clip_or_variant_path(clip: Clip, variant: ClipVariant | None) -> str:
    """返回用于 FFmpeg 的片段路径：优先使用 variant 的 output_path，否则需要从 asset 推导。"""
    if variant and variant.output_path:
        return variant.output_path
    asset = _assets.get(clip.asset_id)
    if asset:
        return asset.source_file_path
    return ""
