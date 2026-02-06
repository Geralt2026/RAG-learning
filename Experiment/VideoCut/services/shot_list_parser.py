"""分镜脚本解析：从 YAML/JSON 加载并校验为 ShotList。"""
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from config.settings import get_settings
from models.shot_list import ShotList, ShotItem, ShotConstraints


def load_shot_list(path: str | Path) -> ShotList:
    """从文件路径加载分镜脚本（支持 .yaml / .yml / .json）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"分镜脚本不存在: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data: dict[str, Any] = yaml.safe_load(raw) or {}
    elif path.suffix.lower() == ".json":
        import json
        data = json.loads(raw)
    else:
        raise ValueError(f"不支持的分镜脚本格式: {path.suffix}")

    return parse_shot_list_dict(data)


def parse_shot_list_dict(data: dict[str, Any]) -> ShotList:
    """将字典解析为 ShotList 模型。"""
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
                    tags=list(c.get("tags", [])) if isinstance(c.get("tags"), list) else [],
                    duration_range=duration_range,
                    shot_type=c.get("shot_type"),
                    resolution=c.get("resolution"),
                )
            else:
                constraints = ShotConstraints()
            shots.append(ShotItem(
                slot=int(s.get("slot", i + 1)),
                type=str(s.get("type", "library")),
                user_slot_id=s.get("user_slot_id"),
                constraints=constraints,
            ))
    return ShotList(
        name=data.get("name", ""),
        version=data.get("version", "1.0"),
        total_duration_target=float(data.get("total_duration_target", 60)),
        shots=shots,
    )


def get_shot_list_by_name(name: str) -> ShotList | None:
    """从默认 shot_lists 目录按名称查找并加载（名称匹配文件名不含后缀）。"""
    settings = get_settings()
    for ext in (".yaml", ".yml", ".json"):
        p = settings.shot_lists_dir / f"{name}{ext}"
        if p.exists():
            return load_shot_list(p)
    return None
