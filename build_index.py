import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))

# 配置
PDF_PATH = os.path.join(current_dir, "data", "pdf", "learn.pdf")
PERSIST_DIR = os.path.join(current_dir, "chroma_db_plan")

# 1. 加载 PDF
print(f"📄 加载 PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"📄 原始页数: {len(docs)}")

# 2. 切片
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✂️ 切片后块数: {len(chunks)}")

# 3. 生成 embedding 并存入 Chroma
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("ZHIPU_API_KEY")
)

print("🔄 正在生成向量库，请稍候...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

print(f"✅ 向量库已生成: {PERSIST_DIR}")
print(f"📊 库内数量: {vectorstore._collection.count()}")