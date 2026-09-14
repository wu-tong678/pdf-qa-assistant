import os
import hashlib
import tempfile
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# 项目根目录（services 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_BASE = os.path.join(BASE_DIR, "chroma_uploads")

# 保证存放目录存在
os.makedirs(CHROMA_BASE, exist_ok=True)


def _get_embeddings():
    return ZhipuAIEmbeddings(
        model="embedding-2",
        api_key=os.getenv("ZHIPU_API_KEY")
    )


def process_pdf(file_content: bytes, file_name: str):
    """
    处理上传的 PDF。
    - file_content: 文件二进制内容
    - file_name: 文件名（仅用于日志，哈希基于内容）
    返回: (vectorstore, chunk_count)
    """
    file_hash = hashlib.md5(file_content).hexdigest()
    persist_dir = os.path.join(CHROMA_BASE, f"chroma_db_{file_hash}")

    embeddings = _get_embeddings()

    # 1. 已存在则直接加载
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectorstore, vectorstore._collection.count()

    # 2. 首次处理
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=BASE_DIR) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 分批插入（智谱 embedding API 单次最多 64 条）
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

        BATCH_SIZE = 64
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                vectorstore.add_documents(batch)
            except Exception:
                # 单批失败则逐条重试
                for doc in batch:
                    try:
                        vectorstore.add_documents([doc])
                    except Exception:
                        pass

        return vectorstore, vectorstore._collection.count()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass