from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    k: int = 3


class ChatResponse(BaseModel):
    answer: str