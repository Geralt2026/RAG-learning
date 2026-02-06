"""视频合成：FFmpeg 拼接、转场、滤镜、字幕。"""
import subprocess
from pathlib import Path
from uuid import uuid4

from config.settings import get_settings
from models.anti_duplicate import RandomParams


def _ffmpeg_cmd() -> list[str]:
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]


def get_video_info(path: str | Path) -> dict | None:
    """获取视频时长、宽高、fps。"""
    path = Path(path)
    if not path.exists():
        return None
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        import json
        data = json.loads(out.stdout)
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        format_info = data.get("format", {})
        duration = float(format_info.get("duration", 0))
        width = int(video_stream.get("width", 1920)) if video_stream else 1920
        height = int(video_stream.get("height", 1080)) if video_stream else 1080
        fps = 0
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


def concat_with_transition(
    segment_paths: list[str],
    output_path: str | Path,
    params: RandomParams,
    transition_duration: float | None = None,
) -> bool:
    """
    将多段视频按顺序拼接，并在段与段之间加入转场。
    segment_paths: 每段视频的本地路径列表。
    """
    if not segment_paths:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    t_dur = transition_duration if transition_duration is not None else params.transition_duration

    # 若仅一段，直接复制并可选缩放/裁剪
    if len(segment_paths) == 1:
        return _apply_single_segment_effects(segment_paths[0], output_path, params)

    # 多段：使用 xfade 或 concat demuxer
    if params.transition_type == "none" or params.transition_type == "fade" and t_dur <= 0:
        # 简单 concat
        list_file = output_path.with_suffix(".txt")
        lines = ["file '" + str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''") + "'" for p in segment_paths]
        list_file.write_text("\n".join(lines), encoding="utf-8")
        cmd = _ffmpeg_cmd() + [
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600, capture_output=True)
            list_file.unlink(missing_ok=True)
            return output_path.exists()
        except subprocess.CalledProcessError:
            list_file.unlink(missing_ok=True)
            return False

    # 带转场的拼接（简化：两两 xfade，或仅首尾淡入淡出）
    # 这里实现一个简化版：先 concat，再整体加淡入淡出
    list_file = output_path.with_suffix(".concat.txt")
    lines = ["file '" + str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''") + "'" for p in segment_paths]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    temp_concat = output_path.parent / f"_temp_{uuid4().hex}.mp4"
    cmd = _ffmpeg_cmd() + [
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vf", f"scale=iw*{params.scale}:ih*{params.scale},crop=iw*{params.crop_ratio}:ih*{params.crop_ratio}",
        "-c:a", "copy", str(temp_concat),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
    except subprocess.CalledProcessError:
        list_file.unlink(missing_ok=True)
        temp_concat.unlink(missing_ok=True)
        return False
    list_file.unlink(missing_ok=True)

    # 淡入淡出
    if params.transition_type in ("fade", "fade_black"):
        info = get_video_info(temp_concat)
        dur = info.get("duration_sec", 10) if info else 10
        fade_in = f"fade=t=in:st=0:d={min(t_dur, dur/4)}"
        fade_out = f"fade=t=out:st={max(0, dur - t_dur)}:d={min(t_dur, dur/4)}"
        cmd2 = _ffmpeg_cmd() + [
            "-i", str(temp_concat),
            "-vf", f"{fade_in},{fade_out}", "-c:a", "copy", str(output_path),
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


def _apply_single_segment_effects(input_path: str, output_path: Path, params: RandomParams) -> bool:
    """单段视频应用缩放/裁剪后输出。"""
    vf = f"scale=iw*{params.scale}:ih*{params.scale},crop=iw*{params.crop_ratio}:ih*{params.crop_ratio}"
    if params.filter_preset == "slight_contrast":
        vf += ",eq=contrast=1.05"
    elif params.filter_preset == "slight_warm":
        vf += ",eq=contrast=1.02:brightness=0.01"
    cmd = _ffmpeg_cmd() + ["-i", input_path, "-vf", vf, "-c:a", "copy", str(output_path)]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False


def burn_subtitles(video_path: str | Path, srt_path: str | Path, output_path: str | Path) -> bool:
    """烧录字幕到视频。"""
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    if not video_path.exists() or not srt_path.exists():
        return False
    # 转义路径中的特殊字符供 filter 使用
    srt_esc = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
    cmd = _ffmpeg_cmd() + [
        "-i", str(video_path), "-vf", f"subtitles='{srt_esc}'", "-c:a", "copy", str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False


def mix_bgm(
    video_path: str | Path,
    bgm_path: str | Path,
    output_path: str | Path,
    bgm_volume: float = 0.3,
) -> bool:
    """混入 BGM（降低 BGM 音量与视频原音混合）。"""
    video_path = Path(video_path)
    bgm_path = Path(bgm_path)
    output_path = Path(output_path)
    if not video_path.exists() or not bgm_path.exists():
        return False
    cmd = _ffmpeg_cmd() + [
        "-i", str(video_path), "-i", str(bgm_path),
        "-filter_complex", f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first",
        "-c:v", "copy", str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False
