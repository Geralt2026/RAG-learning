"""
视频合成编排：解析分镜 -> 选片 -> 随机化 -> 剪辑 -> 成片查重 -> 通过或重试。
使用 LangGraph 风格的状态图（若未安装 langgraph 则用纯 Python 状态机实现）。
"""
from pathlib import Path
from uuid import uuid4
from typing import Any

from config.settings import get_settings
from models.shot_list import ShotList, ShotItem
from models.anti_duplicate import RandomParams, CheckResult
from agent.tools import (
    tool_parse_shot_list,
    tool_pick_clips,
    tool_generate_random_params,
    tool_check_duplicate,
    tool_concat_segments,
    tool_register_fingerprint,
    tool_get_bgm_path,
)
from services.material_library import get_clip_or_variant_path
from services.video_synthesis import mix_bgm


def _run_graph(
    shot_list_path: str,
    user_uploads: dict[str, str] | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    执行完整流程：解析脚本 -> 选片 -> 随机参数 -> 拼接 -> 查重 -> 通过则注册指纹并返回。
    若查重不通过则重试（更换 seed），直到通过或达到最大重试次数。
    """
    user_uploads = user_uploads or {}
    settings = get_settings()
    output_dir = Path(output_dir or settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析分镜
    shot_list = tool_parse_shot_list(shot_list_path)
    if not shot_list:
        return {"success": False, "error": "无法解析分镜脚本", "step": "parse"}

    # 2. 重试循环
    for attempt in range(settings.max_retry_on_duplicate + 1):
        try_seed = (seed + attempt * 1000) if seed is not None else (hash(shot_list_path) + attempt * 1000)
        params = tool_generate_random_params(seed=try_seed)

        # 3. 选片
        segment_paths: list[str] = []
        for shot in shot_list.shots:
            if shot.type == "user_upload" and shot.user_slot_id:
                path = user_uploads.get(shot.user_slot_id)
                if path and Path(path).exists():
                    segment_paths.append(path)
                continue
            if shot.type != "library":
                continue
            picks = tool_pick_clips(shot.constraints, seed=try_seed + shot.slot)
            if not picks:
                # 无可用素材时用占位或跳过（此处跳过该 slot）
                continue
            clip, variant = picks[0]
            p = get_clip_or_variant_path(clip, variant)
            if p and Path(p).exists():
                segment_paths.append(p)

        if not segment_paths:
            return {"success": False, "error": "无可用片段可合成", "step": "pick"}

        # 4. 剪辑
        out_id = uuid4()
        output_path = output_dir / f"output_{out_id}.mp4"
        ok = tool_concat_segments(segment_paths, str(output_path), params)
        if not ok or not output_path.exists():
            continue  # 重试

        # 5. 可选：混 BGM
        bgm = tool_get_bgm_path(params.bgm_index)
        if bgm and bgm.exists():
            with_bgm = output_path.with_stem(output_path.stem + "_bgm")
            if mix_bgm(output_path, bgm, with_bgm, bgm_volume=0.25):
                output_path.unlink(missing_ok=True)
                output_path = with_bgm

        # 6. 成片查重
        result = tool_check_duplicate(str(output_path), threshold=settings.similarity_threshold)
        if result.passed:
            tool_register_fingerprint(out_id, str(output_path))
            return {
                "success": True,
                "output_path": str(output_path),
                "video_id": str(out_id),
                "seed": try_seed,
                "params": params.model_dump(),
                "check_result": result.model_dump(),
            }
        # 不通过则删除本次输出，重试
        output_path.unlink(missing_ok=True)

    return {
        "success": False,
        "error": "达到最大重试次数，成片与历史库相似度过高",
        "step": "check_duplicate",
    }


def run_synthesis(
    shot_list_path: str,
    user_uploads: dict[str, str] | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    对外入口：执行视频合成编排。
    user_uploads: {"intro": "/path/to/intro.mp4"} 等，key 对应分镜中的 user_slot_id。
    """
    return _run_graph(shot_list_path, user_uploads, seed, output_dir)
