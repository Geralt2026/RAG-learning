"""Agent 可调用的工具：素材库选片、防重复检查、提交剪辑。"""
from pathlib import Path
from uuid import UUID

from config.settings import get_settings
from models.asset import Clip, ClipVariant
from models.shot_list import ShotList, ShotItem, ShotConstraints
from models.anti_duplicate import RandomParams, CheckResult
from services.material_library import pick_clips_for_constraints, get_clip_or_variant_path
from services.anti_duplicate import (
    generate_random_params,
    check_duplicate,
    register_fingerprint,
    extract_fingerprint,
)
from services.video_synthesis import (
    concat_with_transition,
    get_video_info,
    burn_subtitles,
    mix_bgm,
)


def tool_parse_shot_list(shot_list_path: str) -> ShotList | None:
    """解析分镜脚本文件，返回 ShotList。"""
    from services.shot_list_parser import load_shot_list
    try:
        return load_shot_list(shot_list_path)
    except Exception:
        return None


def tool_pick_clips(constraints: ShotConstraints, seed: int) -> list[tuple[Clip, ClipVariant | None]]:
    """按约束与种子从素材库选片。"""
    return pick_clips_for_constraints(constraints, seed=seed, prefer_variant=True)


def tool_generate_random_params(seed: int | None = None) -> RandomParams:
    """生成当次合成的随机化参数。"""
    return generate_random_params(seed=seed)


def tool_check_duplicate(
    video_path: str,
    exclude_video_ids: list[str] | None = None,
    threshold: float | None = None,
) -> CheckResult:
    """成片查重。"""
    exclude = [UUID(x) for x in (exclude_video_ids or []) if x]
    return check_duplicate(video_path, exclude_video_ids=exclude or None, threshold=threshold)


def tool_concat_segments(
    segment_paths: list[str],
    output_path: str,
    params: RandomParams,
) -> bool:
    """将多段视频按随机参数拼接。"""
    return concat_with_transition(segment_paths, Path(output_path), params)


def tool_register_fingerprint(video_id: UUID, video_path: str) -> bool:
    """成片入库：提取指纹并注册。"""
    fp = extract_fingerprint(video_path, video_id=video_id)
    if fp:
        register_fingerprint(fp)
        return True
    return False


def tool_get_bgm_path(index: int) -> Path | None:
    """根据索引返回 BGM 路径（若存在）。"""
    s = get_settings()
    if not s.bgm_path.exists():
        return None
    files = sorted(s.bgm_path.glob("*.mp3")) + sorted(s.bgm_path.glob("*.m4a"))
    if 0 <= index < len(files):
        return files[index]
    return files[0] if files else None
