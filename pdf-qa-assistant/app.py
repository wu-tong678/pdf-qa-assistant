#网页界面工具：帮你画出网页上的按钮、输入框、文字显示区域
import streamlit as st
#文本切块工具：把PDF里长长的文字，切成适合向量检索的小块
from langchain_text_splitters import RecursiveCharacterTextSplitter
#向量化工具（智谱版）：调用智谱AI的embedding-2模型，把文字变成向量
from langchain_community.embeddings import ZhipuAIEmbeddings
#大模型工具（智谱版）：调用智谱AI的glm-4-plus模型，生成最终答案
from langchain_community.chat_models import ChatZhipuAI
#向量数据库工具：把向量存到磁盘（./chroma_db），支持持久化检索
from langchain_chroma import Chroma
#RAG问答链工具：把“检索”和“生成”串成一个完整的问答流程
from langchain_classic.chains import RetrievalQA
#PDF加载工具：读取PDF文件内容
from langchain_community.document_loaders import PyPDFLoader
#操作系统接口工具：帮你处理文件路径、检查文件夹是否存在
import os
#哈希计算工具：给文件生成一个独一无二的“指纹”（MD5值），用来区分不同PDF
import hashlib
#临时文件工具：在电脑上创建临时文件，用完自动删除
import tempfile

st.title("📚 PDF智能问答助手")


# ==================== 初始化大模型（只执行一次） ====================
@st.cache_resource
def init_llm():
    """只初始化大模型，向量库每次上传PDF时重新创建"""
    llm = ChatZhipuAI(
        model="glm-4-plus",
        temperature=0,
        api_key=st.secrets["ZHIPU_API_KEY"]  # 替换成你的key
    )
    return llm


# ==================== 处理上传的PDF ====================
def process_pdf(uploaded_file):
    """处理上传的PDF - 支持持久化复用"""

    # 1. 计算文件的唯一标识（基于文件名+内容哈希）
    #.getvalue()_streamlit库中的UploadedFile 类中的方法
    #.getvalue()获取文件原始数据
    file_content = uploaded_file.getvalue()
    #调用模块hashlib里面的函数md5   _Hash 对象的 hexdigest() 方法
    ''''.md5()将文件的二进制内容file_content转化为一个固定长度的哈希对象，
    .hexdigest()
    将这个哈希对象转换为一串由32个十六进制字符（0 - 9
    和a - f）组成的字符串，方便作为文件夹名'''
    file_hash = hashlib.md5(file_content).hexdigest()
    # 用哈希值作为子目录名，区分不同PDF
    persist_dir = f"./chroma_db_{file_hash}"

    embeddings = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key=st.secrets["ZHIPU_API_KEY"]
    )

    # 2. 检查该PDF的向量库是否已存在且是一个文件夹
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        # ✅ 直接加载，跳过向量化
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        # 读取文档数量（从SQLite元数据中获取）
        chunk_count = vectorstore._collection.count()
        return vectorstore, chunk_count

    # 3. 首次处理：向量化并持久化
    try:
        '''tempfile.NamedTemporaryFile()：创建一个临时文件对象。
           delete=False 表示关闭文件后不自动删除；suffix=".pdf" 指定文件扩展名为 .pdf；dir="." 表示在当前目录创建。
           with ... as tmp_file：上下文管理器，确保文件在使用后正确关闭'''
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=".") as tmp_file:
            tmp_file.write(file_content)
            #获取临时文件的路径。
            tmp_path = tmp_file.name

        abs_path = os.path.abspath(tmp_path)
        loader = PyPDFLoader(abs_path)
        #返回一个包含文本内容的 Document 对象列表。
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 创建向量库并持久化
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

        return vectorstore, len(chunks)

    except Exception as e:
        raise Exception(f"PDF处理失败: {str(e)}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass

# ==================== 界面部分 ====================

# 初始化大模型
with st.spinner("初始化模型中..."):
    llm = init_llm()

# 侧边栏：文件上传
with st.sidebar:
    st.header("📄 1. 上传PDF文档")
    uploaded_file = st.file_uploader("选择PDF文件", type=["pdf"])

    st.divider()
    st.header("⚙️ 2. 问答设置")
    k_value = st.slider("检索文档块数量（k值）", 1, 5, 3,
                        help="数值越大，检索到的相关内容越多，但回答可能更冗长")

    st.divider()
    st.header("📖 3. 使用说明")
    st.markdown("""
    1. 上传PDF文件
    2. 等待系统处理（首次需要向量化）
    3. 输入问题
    4. 点击提交
    """)

# 主区域：状态显示
status_placeholder = st.empty()

# 主区域：问答界面
st.subheader("💬 提问")

# 初始化历史记录
if "messages" not in st.session_state:
    #存储历史问答记录
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    #存储向量数据库实例
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:
    #标记PDF是否已处理完成
    st.session_state.pdf_processed = False

# 处理PDF上传
if uploaded_file is not None:
    #当前还没有处理过PDF或者是上传的PDF文件名和上次处理的不一样
    if not st.session_state.pdf_processed or st.session_state.get("last_file_name") != uploaded_file.name:
        with st.spinner(f"正在处理PDF：{uploaded_file.name}..."):
            try:
                #调用函数，分别收到向量数据库对象，文本块数量
                vectorstore, chunk_count = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_processed = True# 标记已完成
                st.session_state.last_file_name = uploaded_file.name# 记录文件名
                status_placeholder.success(f"✅ 已加载PDF：{uploaded_file.name}（共{chunk_count}个文本块）")# 显示成功
            except Exception as e:
                status_placeholder.error(f"❌ 处理PDF失败：{str(e)}") # 显示失败
                st.session_state.pdf_processed = False # 标记未完成
else:
    if st.session_state.pdf_processed:#检查之前是否处理过PDF
        st.session_state.pdf_processed = False#把状态改回"未处理"
        st.session_state.vectorstore = None#清空向量库
        status_placeholder.info("📌 请上传PDF文件开始问答")#.info()显示提示消息

# 输入框和按钮
question = st.text_input("输入你的问题：", disabled=not st.session_state.pdf_processed)

if st.button("提交问题", disabled=not st.session_state.pdf_processed):
    if question:
        with st.spinner("AI思考中..."):
            # 创建RAG问答链
            qa = RetrievalQA.from_chain_type(
                llm=llm,

                #as_retriever()：把向量库转换成检索器，用于搜索相似内容。
                retriever=st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k_value}
                ),
                #返回检索到的原始文档片段
                return_source_documents=True
            )
            #把用户问题传给RAG链执行
            result = qa.invoke({"query": question})
            #从返回结果中提取AI的回答
            answer = result['result']

        # 显示回答
        st.success("回答：")
        #自动判断数据类型，选择合适的显示方式
        st.write(answer)

        # 保存到历史记录
        st.session_state.messages.append({
            "question": question,
            "answer": answer
        })
    else:
        st.warning("请输入问题")

# 显示历史记录
if st.session_state.messages:
    st.divider()
    st.subheader("📝 历史记录")

    # 反向显示，最新的在上面
    for i, msg in enumerate(reversed(st.session_state.messages)):
        with st.container():
            st.write(f"**问：** {msg['question']}")
            st.write(f"**答：** {msg['answer']}")
        if i < len(st.session_state.messages) - 1:
            st.divider()

# 清空历史按钮
if st.session_state.messages:
    if st.button("清空历史记录"):
        st.session_state.messages = []
        st.rerun()
