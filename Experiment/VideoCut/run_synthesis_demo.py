"""
演示：加载示例分镜、注册一条示例素材（需本地有视频文件）、触发生成。
无本地视频时仅演示解析与选片逻辑，不执行 FFmpeg。
"""
from pathlib import Path
import sys

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_settings
from models.asset import Asset, AssetType, AssetMetadata, Clip, ShotType
from models.shot_list import ShotList, ShotConstraints
from services.shot_list_parser import load_shot_list
from services.material_library import add_asset, add_clip, search_clips, pick_clips_for_constraints
from services.anti_duplicate import generate_random_params, get_random_params_space
from agent.graph import run_synthesis


def demo_shot_list():
    """解析并打印分镜脚本。"""
    settings = get_settings()
    yaml_path = settings.shot_lists_dir / "cruise_scenic_v1.yaml"
    if not yaml_path.exists():
        print("未找到 shot_lists/cruise_scenic_v1.yaml，跳过解析演示")
        return
    shot_list = load_shot_list(yaml_path)
    print("分镜脚本:", shot_list.name, shot_list.version)
    for s in shot_list.shots:
        print(f"  slot={s.slot} type={s.type} user_slot_id={s.user_slot_id} constraints.tags={s.constraints.tags}")


def demo_material_library():
    """注册一条示例素材（仅当存在示例视频路径时）。"""
    # 示例路径：可替换为你的本地视频
    demo_video = get_settings().project_root / "demo_cruise.mp4"
    if not demo_video.exists():
        print("无 demo_cruise.mp4，跳过素材库注册；可放一个短视频到项目根后重试")
        return
    asset = Asset(
        type=AssetType.UPLOAD,
        source_file_path=str(demo_video),
        duration_sec=30.0,
        metadata=AssetMetadata(tags=["cruise", "deck", "wide"], shot_type=ShotType.WIDE),
    )
    add_asset(asset)
    clip = Clip(asset_id=asset.id, in_point_sec=0, out_point_sec=10, duration_sec=10)
    add_clip(clip)
    print("已注册示例素材:", asset.id, "Clip:", clip.id)
    picks = pick_clips_for_constraints(ShotConstraints(tags=["cruise", "deck"], duration_range=(5, 15)), seed=42)
    print("按约束选片数量:", len(picks))


def demo_anti_duplicate_params():
    """打印防重复参数空间。"""
    space = get_random_params_space()
    print("防重复参数空间:", space)
    params = generate_random_params(seed=123)
    print("当次随机参数:", params.model_dump())


def demo_run_synthesis():
    """触发生成（无素材时可能选不到片，仅跑通流程）。"""
    settings = get_settings()
    yaml_path = settings.shot_lists_dir / "cruise_scenic_v1.yaml"
    if not yaml_path.exists():
        print("未找到分镜脚本，跳过合成")
        return
    result = run_synthesis(
        shot_list_path=str(yaml_path),
        user_uploads={},
        seed=2025,
    )
    print("合成结果:", result)


if __name__ == "__main__":
    print("=== 1. 分镜脚本解析 ===")
    demo_shot_list()
    print("\n=== 2. 素材库（可选） ===")
    demo_material_library()
    print("\n=== 3. 防重复参数 ===")
    demo_anti_duplicate_params()
    print("\n=== 4. 合成编排（需素材库有可选片段） ===")
    demo_run_synthesis()
