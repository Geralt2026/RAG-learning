"""防重复服务：随机参数生成、帧哈希提取、相似度检测。"""
from pathlib import Path
from uuid import UUID, uuid4
import random

from config.settings import get_settings
from models.anti_duplicate import RandomParams, CheckResult, VideoFingerprint


# 历史成片指纹库（成片 ID -> VideoFingerprint）；生产可改为 DB 或 Redis
_fingerprints: dict[UUID, VideoFingerprint] = {}


def generate_random_params(seed: int | None = None) -> RandomParams:
    """根据种子生成当次合成的随机化参数。"""
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
    """返回当前随机化参数空间，供 API GET /api/anti-duplicate/params。"""
    s = get_settings()
    return {
        "transition_types": s.transition_types,
        "transition_duration_range": list(s.transition_duration_range),
        "scale_range": list(s.scale_range),
        "crop_ratios": s.crop_ratios,
        "similarity_threshold": s.similarity_threshold,
        "max_retry_on_duplicate": s.max_retry_on_duplicate,
    }


def extract_fingerprint(video_path: str | Path, video_id: UUID | None = None) -> VideoFingerprint | None:
    """
    从视频文件提取帧哈希序列作为指纹。
    依赖：opencv-python, imagehash；若未安装则返回 None。
    """
    try:
        import cv2
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
    fp = VideoFingerprint(video_id=vid, frame_hashes=frame_hashes, duration_sec=duration_sec)
    return fp


def _similarity_between_hashes(hashes_a: list[str], hashes_b: list[str]) -> float:
    """计算两段哈希序列的相似度 [0, 1]，基于汉明距离。"""
    try:
        import imagehash
    except ImportError:
        return 0.0

    if not hashes_a or not hashes_b:
        return 0.0
    total = 0.0
    count = 0
    step_a = max(1, len(hashes_a) // 20)
    step_b = max(1, len(hashes_b) // 20)
    for i in range(0, min(len(hashes_a), len(hashes_b)), max(step_a, step_b)):
        ha = imagehash.hex_to_hash(hashes_a[i])
        hb = imagehash.hex_to_hash(hashes_b[min(i * len(hashes_b) // max(1, len(hashes_a)), len(hashes_b) - 1)])
        # 汉明距离 -> 相似度：1 - (dist / 64) 近似
        dist = ha - hb
        total += 1.0 - min(1.0, dist / 64.0)
        count += 1
    return total / count if count else 0.0


def check_duplicate(
    video_path: str | Path,
    exclude_video_ids: list[UUID] | None = None,
    threshold: float | None = None,
) -> CheckResult:
    """
    对待检视频做指纹提取，与历史成片库比对；超过阈值则判定为重复。
    """
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
    """成片入库时调用，将指纹加入历史库。"""
    _fingerprints[fingerprint.video_id] = fingerprint


def unregister_fingerprint(video_id: UUID) -> None:
    """从历史库移除（可选）。"""
    _fingerprints.pop(video_id, None)
