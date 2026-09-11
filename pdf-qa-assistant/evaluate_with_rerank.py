import json
import os
import requests
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
import streamlit as st

# ---------- 配置 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "chroma_db_e860c154de9338af084149f9cecb85a6")

# 加载向量库
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=st.secrets["ZHIPU_API_KEY"]
)
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

# 加载测试集
with open(os.path.join(current_dir, "eval_data.json"), "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 智谱 Rerank 配置
ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
RERANK_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"


# ---------- Rerank 函数 ----------
def rerank(query, documents, top_n=5):
    """调用智谱 Rerank 接口"""
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "rerank",
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    response = requests.post(RERANK_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


# ---------- 评估函数 ----------
def evaluate_with_rerank(k=5, top_n_candidates=20):
    """
    k: 最终返回的文档数
    top_n_candidates: 先由 Chroma 粗排返回的候选数
    """
    hits = 0
    total = len(eval_data)

    for item in eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

        # 1. Chroma 粗排
        candidate_docs = vectorstore.similarity_search(query, k=top_n_candidates)
        candidate_texts = [doc.page_content for doc in candidate_docs]

        # 2. 智谱 Rerank 精排
        rerank_result = rerank(query, candidate_texts, top_n=k)

        # 3. 提取最终结果
        final_docs = []
        for res in rerank_result["results"]:
            idx = res["index"]
            final_docs.append(candidate_docs[idx])

        # 4. 检查命中
        hit = False
        for doc in final_docs:
            if ground_truth.lower() in doc.page_content.lower():
                hit = True
                break

        if hit:
            hits += 1
        else:
            print(f"❌ 未命中: {query}")

    hit_rate = hits / total
    print(f"\n📊 Top-{k} Hit Rate (Chroma + 智谱Rerank): {hit_rate * 100:.2f}% ({hits}/{total})")
    return hit_rate


if __name__ == "__main__":
    print("=" * 50)
    print("Chroma + 智谱 Rerank 评估")
    print("=" * 50)
    evaluate_with_rerank(k=5, top_n_candidates=20)