import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

_llm_instance = None


def init_llm():
    """初始化智谱 GLM-4 大模型（单例）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model="modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            temperature=0
        )
    return _llm_instance