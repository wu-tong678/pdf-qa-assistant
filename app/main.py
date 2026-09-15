from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="PDF QA API",
    description="基于 RAG 的 PDF 智能问答接口",
    version="3.0.0"
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}