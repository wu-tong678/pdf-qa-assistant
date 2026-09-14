import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

_llm_instance = None


def init_llm():
    """初始化智谱 GLM-4 大模型（单例）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatZhipuAI(
            model="glm-4-plus",
            temperature=0,
            api_key=os.getenv("ZHIPU_API_KEY")
        )
    return _llm_instance