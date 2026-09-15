from langchain_classic.chains import RetrievalQA


def ask(vectorstore, llm, question, k=3):
    """
    检索 + 生成答案。
    - vectorstore: Chroma 向量库实例
    - llm: 大模型实例
    - question: 用户问题
    - k: 检索文档块数量
    返回: (answer, source_documents)
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        ),
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    return result["result"], result.get("source_documents", [])
async def ask_stream(vectorstore, llm, question, k=3):
    """
    流式版本：逐字返回答案。
    用 async generator 产出 chunk。
    """
    # 1. 检索
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. 拼 prompt
    prompt = f"""基于以下资料回答问题：
{context}

问题：{question}
"""

    # 3. 流式调用 LLM
    async for chunk in llm.astream(prompt):
        content = chunk.content
        if content:
            yield content