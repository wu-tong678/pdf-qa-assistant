import json
import os
import requests
from collections import defaultdict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- 配置 ----------
PDF_TO_DB = {
    "attention_is_all_you_need.pdf": "chroma_db_attention",
    "glm4_report.pdf": "chroma_db_glm4",
    "langchain_agents.pdf": "chroma_db_langchain",
}

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
RERANK_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"

embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=ZHIPU_API_KEY
)

# ---------- 加载测试集 ----------
with open(os.path.join(current_dir, "data", "v2", "eval_data_v2.json"), "r", encoding="utf-8") as f:
    eval_data = json.load(f)["questions"]


# ---------- 加载三个向量库 ----------
vectorstores = {}
for pdf, db_name in PDF_TO_DB.items():
    db_path = os.path.join(current_dir, db_name)
    vs = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vectorstores[pdf] = vs
    print(f"✅ 加载 {db_name}: {vs._collection.count()} 条")


# ---------- Rerank 函数 ----------
def rerank(query, documents, top_n=5):
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
    response = requests.post(RERANK_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------- 评估 ----------
def evaluate(use_rerank=False, k=5, top_n_candidates=20):
    hits = 0
    total = len(eval_data)
    stats_by_pdf = defaultdict(lambda: {"hit": 0, "total": 0})
    stats_by_diff = defaultdict(lambda: {"hit": 0, "total": 0})

    for item in eval_data:
        query = item["question"]
        ground_truth = item["ground_truth"]
        pdf = item["pdf"]
        diff = item["difficulty"]

        vs = vectorstores[pdf]

        if use_rerank:
            # 1. 粗排
            candidates = vs.similarity_search(query, k=top_n_candidates)
            candidate_texts = [d.page_content for d in candidates]

            # 2. 精排
            try:
                rr = rerank(query, candidate_texts, top_n=k)
                final_docs = [candidates[r["index"]] for r in rr["results"]]
            except Exception as e:
                print(f"⚠️ Rerank 失败，降级用粗排: {e}")
                final_docs = candidates[:k]
        else:
            final_docs = vs.similarity_search(query, k=k)

        # 3. 检查命中
        hit = any(ground_truth.lower() in d.page_content.lower() for d in final_docs)

        if hit:
            hits += 1
            stats_by_pdf[pdf]["hit"] += 1
            stats_by_diff[diff]["hit"] += 1

        stats_by_pdf[pdf]["total"] += 1
        stats_by_diff[diff]["total"] += 1

    # 输出
    mode = "Chroma + Rerank" if use_rerank else "纯 Chroma"
    print(f"\n{'='*50}")
    print(f"📊 {mode} - Top-{k} Hit Rate")
    print(f"{'='*50}")
    print(f"整体: {hits}/{total} = {hits/total*100:.2f}%")

    print(f"\n按 PDF 分:")
    for pdf, s in stats_by_pdf.items():
        rate = s["hit"] / s["total"] * 100 if s["total"] else 0
        print(f"  {pdf}: {s['hit']}/{s['total']} = {rate:.2f}%")

    print(f"\n按难度分:")
    for diff, s in stats_by_diff.items():
        rate = s["hit"] / s["total"] * 100 if s["total"] else 0
        print(f"  {diff}: {s['hit']}/{s['total']} = {rate:.2f}%")

    return hits / total


if __name__ == "__main__":
    print("=" * 50)
    print("v2 评估")
    print("=" * 50)

    print("\n【1】纯 Chroma 评估")
    baseline = evaluate(use_rerank=False, k=5)

    print("\n【2】Chroma + Rerank 评估")
    reranked = evaluate(use_rerank=True, k=5, top_n_candidates=20)

    print("\n" + "=" * 50)
    print("📈 对比总结")
    print("=" * 50)
    print(f"纯 Chroma:        {baseline*100:.2f}%")
    print(f"Chroma + Rerank:  {reranked*100:.2f}%")
    print(f"提升:             {(reranked-baseline)*100:+.2f}%")