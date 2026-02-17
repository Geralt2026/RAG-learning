"""
LangGraph：解析分镜 -> 选片 -> 随机化 -> 剪辑 -> 成片查重 -> 通过或重试。
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from config.settings import get_settings
from models.asset import Asset, Clip, ClipVariant, AssetType, TransformType
from models.shot_list import ShotList, ShotItem, ShotConstraints
from models.anti_duplicate import RandomParams, CheckResult, VideoFingerprint

# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------


class VideoCutState(TypedDict, total=False):
    shot_list_path: str
    user_uploads: dict[str, str]
    output_dir: Path
    max_retry: int
    shot_list: ShotList
    attempt: int
    seed: int
    params: RandomParams
    segment_paths: list[str]
    output_path: str
    video_id: str
    check_result: CheckResult
    error: str
    success: bool
    result: dict[str, Any]


# ---------------------------------------------------------------------------
# 素材库
# ---------------------------------------------------------------------------

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
    _variants_by_clip[clip.id] = list(clip.variant_ids or [])
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


def search_clips(
    tags: list[str] | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
    shot_type: str | None = None,
    asset_type: AssetType | None = None,
    limit: int = 50,
) -> list[Clip]:
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
        if (
            shot_type
            and asset.metadata.shot_type
            and asset.metadata.shot_type.value != shot_type
        ):
            continue
        if tags and not any(t in asset.metadata.tags for t in tags):
            continue
        candidates.append(clip)
    return candidates[:limit]


def pick_clips_for_constraints(
    constraints: ShotConstraints,
    seed: int = 0,
    prefer_variant: bool = True,
) -> list[tuple[Clip, ClipVariant | None]]:
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
    chosen = rng.choice(clips)
    variant = None
    if prefer_variant and chosen.variant_ids:
        vid = rng.choice(chosen.variant_ids)
        variant = _variants.get(vid)
    return [(chosen, variant)]


def get_clip_or_variant_path(clip: Clip, variant: ClipVariant | None) -> str:
    if variant and variant.output_path:
        return variant.output_path
    asset = _assets.get(clip.asset_id)
    if asset:
        return asset.source_file_path
    return ""


# ---------------------------------------------------------------------------
# 分镜解析
# ---------------------------------------------------------------------------


def load_shot_list(path: str | Path) -> ShotList:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"分镜脚本不存在: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data: dict[str, Any] = yaml.safe_load(raw) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"不支持的分镜脚本格式: {path.suffix}")
    return parse_shot_list_dict(data)


def parse_shot_list_dict(data: dict[str, Any]) -> ShotList:
    shots_raw = data.get("shots", [])
    shots: list[ShotItem] = []
    for i, s in enumerate(shots_raw):
        if isinstance(s, dict):
            c = s.get("constraints", {})
            if isinstance(c, dict):
                dr = c.get("duration_range", [3.0, 10.0])
                if isinstance(dr, (list, tuple)) and len(dr) >= 2:
                    duration_range = (float(dr[0]), float(dr[1]))
                else:
                    duration_range = (3.0, 10.0)
                constraints = ShotConstraints(
                    tags=(
                        list(c.get("tags", []))
                        if isinstance(c.get("tags"), list)
                        else []
                    ),
                    duration_range=duration_range,
                    shot_type=c.get("shot_type"),
                    resolution=c.get("resolution"),
                )
            else:
                constraints = ShotConstraints()
            shots.append(
                ShotItem(
                    slot=int(s.get("slot", i + 1)),
                    type=str(s.get("type", "library")),
                    user_slot_id=s.get("user_slot_id"),
                    constraints=constraints,
                )
            )
    return ShotList(
        name=data.get("name", ""),
        version=data.get("version", "1.0"),
        total_duration_target=float(data.get("total_duration_target", 60)),
        shots=shots,
    )


def get_shot_list_by_name(name: str) -> ShotList | None:
    settings = get_settings()
    for ext in (".yaml", ".yml", ".json"):
        p = settings.shot_lists_dir / f"{name}{ext}"
        if p.exists():
            return load_shot_list(p)
    return None


# ---------------------------------------------------------------------------
# 防重复（原 services/anti_duplicate）
# ---------------------------------------------------------------------------

_fingerprints: dict[UUID, VideoFingerprint] = {}


def generate_random_params(seed: int | None = None) -> RandomParams:
    s = get_settings()
    rng = random.Random(seed)
    transition = rng.choice(s.transition_types)
    t_min, t_max = s.transition_duration_range
    scale_min, scale_max = s.scale_range
    return RandomParams(
        seed=seed or rng.randint(0, 2**31 - 1),
        transition_type=transition,
        transition_duration=round(rng.uniform(t_min, t_max), 2),
        scale=round(rng.uniform(scale_min, scale_max), 3),
        crop_ratio=rng.choice(s.crop_ratios),
        filter_preset=rng.choice(["none", "slight_contrast", "slight_warm"]),
        bgm_index=rng.randint(0, 99),
    )


def get_random_params_space() -> dict:
    s = get_settings()
    return {
        "transition_types": s.transition_types,
        "transition_duration_range": list(s.transition_duration_range),
        "scale_range": list(s.scale_range),
        "crop_ratios": s.crop_ratios,
        "similarity_threshold": s.similarity_threshold,
        "max_retry_on_duplicate": s.max_retry_on_duplicate,
    }


def extract_fingerprint(
    video_path: str | Path, video_id: UUID | None = None
) -> VideoFingerprint | None:
    try:
        # cv2 是 OpenCV 的 Python 接口库，常用于计算机视觉与处理图片或视频帧。
        import cv2  # OpenCV，用于视频帧读取和处理
        import imagehash
        from PIL import Image
    except ImportError:
        return None
    path = Path(video_path)
    if not path.exists():
        return None
    settings = get_settings()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    interval_frames = max(1, int(fps * settings.frame_sample_interval_sec))
    frame_hashes: list[str] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h = imagehash.phash(pil, hash_size=settings.hash_size)
            frame_hashes.append(str(h))
        frame_idx += 1
    cap.release()
    vid = video_id or uuid4()
    return VideoFingerprint(
        video_id=vid, frame_hashes=frame_hashes, duration_sec=duration_sec
    )


def _similarity_between_hashes(hashes_a: list[str], hashes_b: list[str]) -> float:
    try:
        import imagehash
    except ImportError:
        return 0.0
    if not hashes_a or not hashes_b:
        return 0.0
    total, count = 0.0, 0
    step = max(1, min(len(hashes_a), len(hashes_b)) // 20)
    for i in range(0, min(len(hashes_a), len(hashes_b)), step):
        ha = imagehash.hex_to_hash(hashes_a[i])
        j = min(i * len(hashes_b) // max(1, len(hashes_a)), len(hashes_b) - 1)
        hb = imagehash.hex_to_hash(hashes_b[j])
        total += 1.0 - min(1.0, (ha - hb) / 64.0)
        count += 1
    return total / count if count else 0.0


def check_duplicate(
    video_path: str | Path,
    exclude_video_ids: list[UUID] | None = None,
    threshold: float | None = None,
) -> CheckResult:
    settings = get_settings()
    thr = threshold if threshold is not None else settings.similarity_threshold
    exclude = set(exclude_video_ids or [])
    fp = extract_fingerprint(video_path)
    if not fp or not fp.frame_hashes:
        return CheckResult(passed=True, message="无法提取指纹，跳过查重")
    similar_ids: list[UUID] = []
    max_sim = 0.0
    for vid, existing in _fingerprints.items():
        if vid in exclude:
            continue
        sim = _similarity_between_hashes(fp.frame_hashes, existing.frame_hashes)
        if sim > max_sim:
            max_sim = sim
        if sim >= thr:
            similar_ids.append(vid)
    passed = max_sim < thr
    return CheckResult(
        passed=passed,
        similar_video_ids=similar_ids,
        max_similarity=round(max_sim, 4),
        message="通过" if passed else f"与 {len(similar_ids)} 条历史成片过于相似",
    )


def register_fingerprint(fingerprint: VideoFingerprint) -> None:
    _fingerprints[fingerprint.video_id] = fingerprint


# ---------------------------------------------------------------------------
# 视频合成
# ---------------------------------------------------------------------------


def _ffmpeg_cmd() -> list[str]:
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]


def get_video_info(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
        )
        format_info = data.get("format", {})
        duration = float(format_info.get("duration", 0))
        width = int(video_stream.get("width", 1920)) if video_stream else 1920
        height = int(video_stream.get("height", 1080)) if video_stream else 1080
        fps = 25.0
        if video_stream and "r_frame_rate" in video_stream:
            r = video_stream["r_frame_rate"]
            if "/" in r:
                a, b = r.split("/")
                fps = float(a) / float(b) if float(b) else 25.0
            else:
                fps = float(r)
        return {"duration_sec": duration, "width": width, "height": height, "fps": fps}
    except Exception:
        return None


def _apply_single_segment_effects(
    input_path: str, output_path: Path, params: RandomParams
) -> bool:
    vf = f"scale=iw*{params.scale}:ih*{params.scale},crop=iw*{params.crop_ratio}:ih*{params.crop_ratio}"
    if params.filter_preset == "slight_contrast":
        vf += ",eq=contrast=1.05"
    elif params.filter_preset == "slight_warm":
        vf += ",eq=contrast=1.02:brightness=0.01"
    cmd = _ffmpeg_cmd() + [
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:a",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False


def concat_with_transition(
    segment_paths: list[str],
    output_path: str | Path,
    params: RandomParams,
    transition_duration: float | None = None,
) -> bool:
    if not segment_paths:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    t_dur = (
        transition_duration
        if transition_duration is not None
        else params.transition_duration
    )
    if len(segment_paths) == 1:
        return _apply_single_segment_effects(segment_paths[0], output_path, params)
    if params.transition_type == "none" or (
        params.transition_type == "fade" and t_dur <= 0
    ):
        list_file = output_path.with_suffix(".txt")
        lines = [
            "file '"
            + str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''")
            + "'"
            for p in segment_paths
        ]
        list_file.write_text("\n".join(lines), encoding="utf-8")
        cmd = _ffmpeg_cmd() + [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600, capture_output=True)
            list_file.unlink(missing_ok=True)
            return output_path.exists()
        except subprocess.CalledProcessError:
            list_file.unlink(missing_ok=True)
            return False
    list_file = output_path.with_suffix(".concat.txt")
    lines = [
        "file '" + str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''") + "'"
        for p in segment_paths
    ]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    temp_concat = output_path.parent / f"_temp_{uuid4().hex}.mp4"
    cmd = _ffmpeg_cmd() + [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"scale=iw*{params.scale}:ih*{params.scale},crop=iw*{params.crop_ratio}:ih*{params.crop_ratio}",
        "-c:a",
        "copy",
        str(temp_concat),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
    except subprocess.CalledProcessError:
        list_file.unlink(missing_ok=True)
        temp_concat.unlink(missing_ok=True)
        return False
    list_file.unlink(missing_ok=True)
    if params.transition_type in ("fade", "fade_black"):
        info = get_video_info(temp_concat)
        dur = info.get("duration_sec", 10) if info else 10
        fade_in = f"fade=t=in:st=0:d={min(t_dur, dur/4)}"
        fade_out = f"fade=t=out:st={max(0, dur - t_dur)}:d={min(t_dur, dur/4)}"
        cmd2 = _ffmpeg_cmd() + [
            "-i",
            str(temp_concat),
            "-vf",
            f"{fade_in},{fade_out}",
            "-c:a",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(cmd2, check=True, timeout=300, capture_output=True)
        except subprocess.CalledProcessError:
            pass
        temp_concat.unlink(missing_ok=True)
        return output_path.exists()
    import shutil

    shutil.move(str(temp_concat), str(output_path))
    return output_path.exists()


def mix_bgm(
    video_path: str | Path,
    bgm_path: str | Path,
    output_path: str | Path,
    bgm_volume: float = 0.3,
) -> bool:
    video_path = Path(video_path)
    bgm_path = Path(bgm_path)
    output_path = Path(output_path)
    if not video_path.exists() or not bgm_path.exists():
        return False
    cmd = _ffmpeg_cmd() + [
        "-i",
        str(video_path),
        "-i",
        str(bgm_path),
        "-filter_complex",
        f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first",
        "-c:v",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False


def get_bgm_path(index: int) -> Path | None:
    s = get_settings()
    if not s.bgm_path.exists():
        return None
    files = sorted(s.bgm_path.glob("*.mp3")) + sorted(s.bgm_path.glob("*.m4a"))
    if 0 <= index < len(files):
        return files[index]
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Clip 变体任务
# ---------------------------------------------------------------------------


def generate_variant(
    clip: Clip,
    asset_source_path: str,
    transform: TransformType,
    output_dir: Path | None = None,
) -> ClipVariant | None:
    settings = get_settings()
    output_dir = output_dir or settings.clips_path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{clip.id}_{transform.value}_{uuid4().hex[:8]}.mp4"
    out_path = output_dir / out_name
    in_point, out_point = clip.in_point_sec, clip.out_point_sec
    duration = clip.duration_sec
    base = ["-ss", str(in_point), "-t", str(duration), "-i", str(asset_source_path)]
    vf: list[str] = []
    if transform == TransformType.MIRROR_H:
        vf.append("hflip")
    elif transform == TransformType.MIRROR_V:
        vf.append("vflip")
    elif transform == TransformType.SPEED_09:
        vf.append("setpts=1.11*PTS")
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
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + base
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    except subprocess.CalledProcessError:
        return None
    return ClipVariant(
        clip_id=clip.id,
        transform=transform,
        output_path=str(out_path),
        duration_sec=duration,
    )


def generate_all_variants_for_clip(
    clip: Clip,
    asset_source_path: str,
    transforms: list[TransformType] | None = None,
) -> list[ClipVariant]:
    transforms = transforms or [
        TransformType.ORIGINAL,
        TransformType.MIRROR_H,
        TransformType.SPEED_105,
        TransformType.CROP_CENTER_98,
    ]
    results: list[ClipVariant] = []
    for t in transforms:
        if t == TransformType.ORIGINAL:
            results.append(
                ClipVariant(
                    clip_id=clip.id,
                    transform=t,
                    output_path=asset_source_path,
                    duration_sec=clip.duration_sec,
                )
            )
            continue
        v = generate_variant(clip, asset_source_path, t)
        if v:
            results.append(v)
    return results


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------


def _node_parse(state: VideoCutState) -> dict[str, Any]:
    try:
        shot_list = load_shot_list(state["shot_list_path"])
        return {"shot_list": shot_list, "attempt": 0, "error": None}
    except Exception as e:
        return {
            "success": False,
            "error": "无法解析分镜脚本",
            "result": {"success": False, "error": str(e), "step": "parse"},
        }


def _node_gen_params(state: VideoCutState) -> dict[str, Any]:
    attempt = state.get("attempt", 0)
    base_seed = state.get("seed")
    shot_list_path = state.get("shot_list_path", "")
    try_seed = (
        (base_seed + attempt * 1000)
        if base_seed is not None
        else (hash(shot_list_path) + attempt * 1000)
    )
    params = generate_random_params(seed=try_seed)
    return {"seed": try_seed, "params": params}


def _node_pick(state: VideoCutState) -> dict[str, Any]:
    shot_list = state["shot_list"]
    user_uploads = state.get("user_uploads") or {}
    try_seed = state["seed"]
    segment_paths: list[str] = []
    for shot in shot_list.shots:
        if shot.type == "user_upload" and shot.user_slot_id:
            path = user_uploads.get(shot.user_slot_id)
            if path and Path(path).exists():
                segment_paths.append(path)
            continue
        if shot.type != "library":
            continue
        picks = pick_clips_for_constraints(shot.constraints, seed=try_seed + shot.slot)
        if not picks:
            continue
        clip, variant = picks[0]
        p = get_clip_or_variant_path(clip, variant)
        if p and Path(p).exists():
            segment_paths.append(p)
    if not segment_paths:
        return {
            "success": False,
            "error": "无可用片段可合成",
            "result": {"success": False, "error": "无可用片段可合成", "step": "pick"},
        }
    return {"segment_paths": segment_paths}


def _node_concat(state: VideoCutState) -> dict[str, Any]:
    segment_paths = state["segment_paths"]
    output_dir = state["output_dir"]
    params = state["params"]
    out_id = uuid4()
    output_path = output_dir / f"output_{out_id}.mp4"
    ok = concat_with_transition(segment_paths, str(output_path), params)
    if not ok or not output_path.exists():
        return {
            "success": False,
            "error": "剪辑失败",
            "result": {"success": False, "error": "剪辑失败", "step": "concat"},
        }
    return {"output_path": str(output_path), "video_id": str(out_id)}


def _node_mix_bgm(state: VideoCutState) -> dict[str, Any]:
    output_path = state["output_path"]
    params = state["params"]
    bgm = get_bgm_path(params.bgm_index)
    if bgm and bgm.exists():
        with_bgm = Path(output_path).with_stem(Path(output_path).stem + "_bgm")
        if mix_bgm(output_path, bgm, with_bgm, bgm_volume=0.25):
            Path(output_path).unlink(missing_ok=True)
            return {"output_path": str(with_bgm)}
    return {}


def _node_check_duplicate(state: VideoCutState) -> dict[str, Any]:
    settings = get_settings()
    result = check_duplicate(
        state["output_path"], threshold=settings.similarity_threshold
    )
    return {"check_result": result}


def _node_register(state: VideoCutState) -> dict[str, Any]:
    from uuid import UUID

    vid = UUID(state["video_id"])
    fp = extract_fingerprint(state["output_path"], video_id=vid)
    if fp:
        register_fingerprint(fp)
    return {}


def _node_fail_result(state: VideoCutState) -> dict[str, Any]:
    if state.get("result") is not None:
        return {}
    return {
        "result": {
            "success": False,
            "error": state.get("error") or "达到最大重试次数，成片与历史库相似度过高",
            "step": "check_duplicate",
        }
    }


def _node_success_result(state: VideoCutState) -> dict[str, Any]:
    return {
        "result": {
            "success": True,
            "output_path": state["output_path"],
            "video_id": state["video_id"],
            "seed": state["seed"],
            "params": state["params"].model_dump(),
            "check_result": state["check_result"].model_dump(),
        }
    }


def _after_check(state: VideoCutState) -> str:
    if state["check_result"].passed:
        return "register"
    attempt = state.get("attempt", 0)
    max_retry = state.get("max_retry", 3)
    if attempt < max_retry:
        return "retry"
    return "fail"


# ---------------------------------------------------------------------------
# 建图与入口
# ---------------------------------------------------------------------------


def _build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(VideoCutState)

    graph.add_node("parse", _node_parse)
    graph.add_node("gen_params", _node_gen_params)
    graph.add_node("pick", _node_pick)
    graph.add_node("concat", _node_concat)
    graph.add_node("mix_bgm", _node_mix_bgm)
    graph.add_node("check_duplicate", _node_check_duplicate)
    graph.add_node("register", _node_register)
    graph.add_node("success_result", _node_success_result)
    graph.add_node("fail_result", _node_fail_result)
    graph.add_node("increment_retry", lambda s: {"attempt": s.get("attempt", 0) + 1})

    graph.set_entry_point("parse")

    def route_after_parse(state: VideoCutState) -> str:
        if state.get("result") is not None:
            return "fail_result"
        return "gen_params"

    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {"gen_params": "gen_params", "fail_result": "fail_result"},
    )
    graph.add_edge("gen_params", "pick")

    def route_after_pick(state: VideoCutState) -> str:
        if state.get("result") is not None:
            return "fail_result"
        return "concat"

    graph.add_conditional_edges(
        "pick", route_after_pick, {"concat": "concat", "fail_result": "fail_result"}
    )

    def route_after_concat(state: VideoCutState) -> str:
        if state.get("result") is not None:
            return "fail_result"
        return "mix_bgm"

    graph.add_conditional_edges(
        "concat",
        route_after_concat,
        {"mix_bgm": "mix_bgm", "fail_result": "fail_result"},
    )
    graph.add_edge("mix_bgm", "check_duplicate")
    graph.add_conditional_edges(
        "check_duplicate",
        _after_check,
        {"register": "register", "retry": "increment_retry", "fail": "fail_result"},
    )
    graph.add_edge("increment_retry", "gen_params")
    graph.add_edge("register", "success_result")
    graph.add_edge("success_result", END)
    graph.add_edge("fail_result", END)

    return graph.compile()


# 延迟编译，避免顶层 import langgraph 失败时影响其他导入
_synth_graph = None


def _get_graph():
    global _synth_graph
    if _synth_graph is None:
        _synth_graph = _build_graph()
    return _synth_graph


def run_synthesis(
    shot_list_path: str,
    user_uploads: dict[str, str] | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """对外入口：执行视频合成编排（LangGraph）。"""
    user_uploads = user_uploads or {}
    settings = get_settings()
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initial: VideoCutState = {
        "shot_list_path": shot_list_path,
        "user_uploads": user_uploads,
        "output_dir": out_dir,
        "max_retry": settings.max_retry_on_duplicate,
        "seed": seed,
    }
    result_state = _get_graph().invoke(initial)
    result = result_state.get("result")
    if result is not None:
        return result
    return {
        "success": False,
        "error": result_state.get("error", "未知错误"),
        "step": "unknown",
    }
