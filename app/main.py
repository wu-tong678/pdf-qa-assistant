import time
from fastapi import FastAPI, Request
from app.api.routes import router
from core.logging import logger

app = FastAPI(
    title="PDF QA API",
    description="基于 RAG 的 PDF 智能问答接口",
    version="3.0.0"
)

app.include_router(router, prefix="/api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的方法、路径、状态码、耗时"""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} | "
        f"{response.status_code} | {elapsed:.2f}s"
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}