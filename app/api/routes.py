import os
from fastapi import APIRouter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings

from app.schemas.chat import ChatRequest, ChatResponse
from services.llm import init_llm
from services.rag import ask

load_dotenv()

router = APIRouter()

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 懒加载全局资源
_vectorstore = None
_llm = None


def _get_resources():
    """懒加载 LLM 和向量库"""
    global _vectorstore, _llm
    if _vectorstore is None:
        embeddings = ZhipuAIEmbeddings(
            model="embedding-2",
            api_key=os.getenv("ZHIPU_API_KEY")
        )
        _vectorstore = Chroma(
            persist_directory=os.path.join(BASE_DIR, "chroma_db_attention"),
            embedding_function=embeddings
        )
    if _llm is None:
        _llm = init_llm()
    return _vectorstore, _llm


@router.get("/ping")
def ping():
    return {"message": "pong"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    vectorstore, llm = _get_resources()
    answer, _ = ask(vectorstore, llm, req.question, k=req.k)
    return ChatResponse(answer=answer)