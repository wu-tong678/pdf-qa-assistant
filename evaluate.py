import json
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
import streamlit as st

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载向量库（使用你实际存在的哈希文件夹）
persist_dir = os.path.join(current_dir, "chroma_db_e860c154de9338af084149f9cecb85a6")
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=st.secrets["ZHIPU_API_KEY"]
)

# 加载向量库
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

# 加载测试集
eval_data_path = os.path.join(current_dir, "eval_data.json")
with open(eval_data_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)


def evaluate(k=5):
    """评估检索效果，计算Top-k命中率"""
    if not eval_data:
        print("测试集为空，请填充 eval_data.json")
        return

    hits = 0
    total = len(eval_data)

    for item in eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

        # 检索
        docs = vectorstore.similarity_search(query, k=k)

        # 检查是否命中（ground_truth的关键词出现在检索结果中）
        hit = False
        for doc in docs:
            if ground_truth.lower() in doc.page_content.lower():
                hit = True
                break

        if hit:
            hits += 1
        else:
            print(f"❌ 未命中: {query} -> 期望包含 '{ground_truth}'")

    hit_rate = hits / total
    print(f"\n📊 Top-{k} Hit Rate: {hit_rate * 100:.2f}% ({hits}/{total})")
    return hit_rate


if __name__ == "__main__":
    evaluate(k=5)