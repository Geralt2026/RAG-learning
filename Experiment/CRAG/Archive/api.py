# -*- coding: utf-8 -*-
"""
FastAPI 接口（PydanticAI 版）：问答与知识库重建
"""
import os
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crag import run_crag

CRAG_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="CRAG 知识库问答",
    description="基于 PydanticAI + Corrective RAG 的知识库问答 API",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")


class AskResponse(BaseModel):
    answer: str = Field(..., description="生成回答")
    action: str = Field(..., description="correct / incorrect / ambiguous")
    context_used: str = Field("", description="实际使用的参考资料摘要")
    num_retrieved: int = Field(0, description="检索到的文档数")


class BuildKBResponse(BaseModel):
    message: str = Field(..., description="构建结果说明")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """提交问题，返回 CRAG 生成的回答及动作类型（内部使用 PydanticAI）。"""
    try:
        result = run_crag(req.question)
        return AskResponse(
            answer=result["answer"],
            action=result["action"],
            context_used=result.get("context_used", ""),
            num_retrieved=result.get("num_retrieved", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build_kb", response_model=BuildKBResponse)
def build_kb(force_rebuild: bool = False):
    """使用 MinerU 从 Files 目录加载 PDF 并构建向量库（子进程执行）。"""
    try:
        code = (
            "from knowledge_base import build_knowledge_base; "
            "print(build_knowledge_base(force_rebuild=%s))" % str(force_rebuild)
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([CRAG_DIR, os.environ.get("PYTHONPATH", "")])}
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=CRAG_DIR,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="构建失败（exit %s）：%s\n%s" % (result.returncode, stdout or "(无输出)", stderr),
            )
        return BuildKBResponse(message=stdout or "构建完成。")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="构建超时，请稍后重试或改用命令行构建。")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
