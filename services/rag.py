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