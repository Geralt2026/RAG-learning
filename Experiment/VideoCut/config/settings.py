"""应用配置。"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，可从环境变量或 .env 读取。"""

    # 项目路径
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    shot_lists_dir: Path = Path(__file__).resolve().parent.parent / "shot_lists"
    output_dir: Path = Path(__file__).resolve().parent.parent / "output"

    # 素材库（本地演示用目录；生产可改为 OSS path）
    storage_path: Path = Path(__file__).resolve().parent.parent / "storage"
    assets_path: Path = Path(__file__).resolve().parent.parent / "storage" / "assets"
    clips_path: Path = Path(__file__).resolve().parent.parent / "storage" / "clips"
    bgm_path: Path = Path(__file__).resolve().parent.parent / "storage" / "bgm"

    # 防重复
    similarity_threshold: float = 0.85  # 超过此相似度判定为重复，触发重生成
    frame_sample_interval_sec: float = 1.0  # 帧采样间隔（秒）
    max_retry_on_duplicate: int = 3  # 相似度过高时最大重试次数
    hash_size: int = 16  # 感知哈希尺寸（pHash/dHash）

    # 视频合成
    default_width: int = 1920
    default_height: int = 1080
    default_fps: float = 25.0

    # 随机化参数空间（供 GET /api/anti-duplicate/params 等使用）
    transition_types: list[str] = ["none", "fade", "fade_black", "wipe"]
    transition_duration_range: tuple[float, float] = (0.3, 0.8)
    scale_range: tuple[float, float] = (1.0, 1.05)
    crop_ratios: list[float] = [1.0, 0.98, 0.95]

    # 数据库（可选，演示可用 SQLite）
    database_url: str = "sqlite:///./qmt.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
