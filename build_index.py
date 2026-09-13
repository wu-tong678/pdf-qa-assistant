import sys
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 3:
    print("用法: python build_index.py <pdf路径> <向量库目录名>")
    sys.exit(1)

pdf_path = os.path.join(current_dir, sys.argv[1])
persist_dir = os.path.join(current_dir, sys.argv[2])

print(f"📄 加载 PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"📄 原始页数: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 过滤异常 chunk
chunks = [
    c for c in chunks
    if c.page_content and len(c.page_content.strip()) > 10
]
print(f"✂️ 切片后块数: {len(chunks)}")

embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("ZHIPU_API_KEY")
)

# 初始化空向量库
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# 分批插入，每批最多 64 条
BATCH_SIZE = 64
total = len(chunks)
success = 0
failed = 0

print(f"🔄 分批生成向量库，每批 {BATCH_SIZE} 条...")
for i in range(0, total, BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    try:
        vectorstore.add_documents(batch)
        success += len(batch)
        print(f"  ✅ 已完成 {min(i + BATCH_SIZE, total)}/{total}")
    except Exception as e:
        # 一批失败，逐条重试，单条失败的跳过
        print(f"  ⚠️ 批次 {i} 失败，逐条重试...")
        for doc in batch:
            try:
                vectorstore.add_documents([doc])
                success += 1
            except Exception as e2:
                failed += 1
                print(f"    ❌ 跳过异常 chunk: {str(e2)[:80]}")

print(f"✅ 成功: {success}, 失败/跳过: {failed}")
print(f"✅ 向量库已生成: {persist_dir}")
print(f"📊 库内数量: {vectorstore._collection.count()}")