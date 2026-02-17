"""FastAPI 入口：素材库、防重复、合成触发等接口。"""
import sys
from pathlib import Path

# 保证从任意工作目录启动时都能找到 config、agent 等包（项目根 = VideoCut）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from uuid import UUID

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import get_settings
from agent.graph import (
    run_synthesis,
    check_duplicate,
    get_random_params_space,
)


app = FastAPI(title="视频合成与防重复", version="1.0.0")


# ---------- 防重复 ----------

class CheckRequest(BaseModel):
    video_path: str | None = None  # 服务端路径
    exclude_video_ids: list[str] = []
    threshold: float | None = None


class CheckResponse(BaseModel):
    passed: bool
    similar_video_ids: list[str]
    max_similarity: float
    message: str


@app.post("/api/anti-duplicate/check", response_model=CheckResponse)
def api_check_duplicate(req: CheckRequest):
    """成片查重。"""
    if not req.video_path or not Path(req.video_path).exists():
        raise HTTPException(400, "video_path 无效或文件不存在")
    exclude = [UUID(x) for x in req.exclude_video_ids if x]
    result = check_duplicate(req.video_path, exclude_video_ids=exclude, threshold=req.threshold)
    return CheckResponse(
        passed=result.passed,
        similar_video_ids=[str(x) for x in result.similar_video_ids],
        max_similarity=result.max_similarity,
        message=result.message,
    )


@app.get("/api/anti-duplicate/params")
def api_get_params():
    """返回当前随机化参数空间与查重阈值。"""
    return get_random_params_space()


# ---------- 合成触发 ----------

class SynthesisRequest(BaseModel):
    shot_list_path: str
    seed: int | None = None
    # user_uploads 通过 multipart 上传时用 form 的 key 对应 user_slot_id


@app.post("/api/synthesis/run")
def api_run_synthesis(
    shot_list_path: str = Form(...),
    seed: str | None = Form(None),
):
    """根据分镜脚本触发生成；用户片段需先上传到临时目录再在 shot_list 中引用路径。"""
    path = Path(shot_list_path)
    if not path.is_absolute():
        path = get_settings().shot_lists_dir / shot_list_path
    if not path.exists():
        path = Path(shot_list_path)
    if not path.exists():
        raise HTTPException(400, "分镜脚本文件不存在")
    seed_int = int(seed) if seed is not None and str(seed).strip().lstrip("-").isdigit() else None
    result = run_synthesis(str(path), user_uploads={}, seed=seed_int)
    return result


# ---------- 健康检查 ----------

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)