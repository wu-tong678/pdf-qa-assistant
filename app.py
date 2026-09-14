import streamlit as st
from services.rag import ask
from services.llm import init_llm
from services.pdf import process_pdf

st.title("📚 PDF智能问答助手")

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
                vectorstore, chunk_count = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
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
            answer, _ = ask(st.session_state.vectorstore, llm, question, k=k_value)

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
