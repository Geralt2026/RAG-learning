"""
异步任务：为 Clip 生成多版本二次剪辑（镜像、变速、裁剪）。
生产环境可接入 Celery/Dramatiq，此处提供可调用的函数接口。
"""
from pathlib import Path
from uuid import uuid4
import subprocess

from config.settings import get_settings
from models.asset import Clip, ClipVariant, TransformType


def _ffmpeg(args: list[str], timeout: int = 300) -> bool:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def generate_variant(
    clip: Clip,
    asset_source_path: str,
    transform: TransformType,
    output_dir: Path | None = None,
) -> ClipVariant | None:
    """
    为一条 Clip 生成指定变换的 ClipVariant，写入 output_dir 并返回模型。
    """
    settings = get_settings()
    output_dir = output_dir or settings.clips_path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{clip.id}_{transform.value}_{uuid4().hex[:8]}.mp4"
    out_path = output_dir / out_name

    in_point = clip.in_point_sec
    out_point = clip.out_point_sec
    duration = clip.duration_sec

    base = [
        "-ss", str(in_point), "-t", str(duration),
        "-i", str(asset_source_path),
    ]
    vf: list[str] = []

    if transform == TransformType.MIRROR_H:
        vf.append("hflip")
    elif transform == TransformType.MIRROR_V:
        vf.append("vflip")
    elif transform == TransformType.SPEED_09:
        vf.append("setpts=1.11*PTS")  # 0.9x
    elif transform == TransformType.SPEED_105:
        vf.append("setpts=0.952*PTS")
    elif transform == TransformType.SPEED_11:
        vf.append("setpts=0.909*PTS")
    elif transform in (TransformType.CROP_CENTER_98, TransformType.CROP_CENTER_95):
        r = 0.98 if transform == TransformType.CROP_CENTER_98 else 0.95
        vf.append(f"crop=iw*{r}:ih*{r}:(iw-iw*{r})/2:(ih-ih*{r})/2,scale=iw:ih")

    if vf:
        base += ["-vf", ",".join(vf)]
    base += ["-c:a", "copy", str(out_path)]

    if not _ffmpeg(base):
        return None

    variant = ClipVariant(
        clip_id=clip.id,
        transform=transform,
        output_path=str(out_path),
        duration_sec=duration,
    )
    return variant


def generate_all_variants_for_clip(
    clip: Clip,
    asset_source_path: str,
    transforms: list[TransformType] | None = None,
) -> list[ClipVariant]:
    """为一条 Clip 批量生成多种变换版本。"""
    from models.asset import TransformType as TT
    transforms = transforms or [
        TT.ORIGINAL,
        TT.MIRROR_H,
        TT.SPEED_105,
        TT.CROP_CENTER_98,
    ]
    results: list[ClipVariant] = []
    for t in transforms:
        if t == TT.ORIGINAL:
            # original 可直接用源路径，不生成新文件
            v = ClipVariant(clip_id=clip.id, transform=t, output_path=asset_source_path, duration_sec=clip.duration_sec)
            results.append(v)
            continue
        v = generate_variant(clip, asset_source_path, t)
        if v:
            results.append(v)
    return results
