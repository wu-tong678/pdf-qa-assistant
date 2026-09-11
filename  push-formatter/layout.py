import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(page_title="推送排版小助手", page_icon="📝", layout="centered")

# 标题
st.title("📝 推送排版小助手")
st.markdown("> 请输入活动主题呀，小小AI帮你生成标题和正文大纲")

# 侧边栏说明
with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown("1. 输入让你头疼的主题/内容")
    st.markdown("2. 马上🐴点击「生成标题」")
    st.markdown("3. 选择喜欢且引人的标题")
    st.markdown("4. 点击「生成正文大纲」,马上呈现")
    st.markdown("5. 我一键就复制到公众号")

    st.markdown("---")
    st.markdown("### 看这里，紧急提示")
    st.markdown("- 标题越具体，效果越好，让小手手动一动❤️")
    st.markdown("- 可以描述活动时间、地点、亮点")

# 你的智谱API Key（替换成你自己的）
API_KEY = st.secrets["ZHIPU_API_KEY"]


# 调用大模型的函数
def call_glm(prompt):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8
    }
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"请求失败：{response.status_code}"


# 初始化session_state
if "generated_titles" not in st.session_state:
    st.session_state.generated_titles = []
if "selected_title" not in st.session_state:
    st.session_state.selected_title = ""
if "generated_outline" not in st.session_state:
    st.session_state.generated_outline = ""

# 主界面输入
activity = st.text_area(
    "📌 奇思妙想起来啊",
    placeholder="例如：主题：魔仙堡大会\n时间：6月15日 19:00\n地点：荷花池底，嘿嘿",
    height=120
)

# 第一行按钮
col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("✨ 生成标题", type="primary", use_container_width=True)

# 生成标题
if generate_btn and activity:
    with st.spinner("AI正在为主人拼命中..."):
        prompt = f"""你是一个公众号推送标题专家。根据以下活动信息，生成5个吸引人的推送标题。
要求：
1. 标题要吸引人点击
2. 风格可以不同：正式的、有趣的、悬念的、直白的
3. 每个标题单独一行

活动信息：
{activity}

请直接输出5个标题，每行一个，不要加序号。"""

        result = call_glm(prompt)
        if "请求失败" not in result:
            st.session_state.generated_titles = result.strip().split("\n")
            st.session_state.generated_outline = ""  # 清空旧的大纲
            st.session_state.selected_title = ""
        else:
            st.error(result)

# 显示生成的标题
if st.session_state.generated_titles:
    st.markdown("---")
    st.subheader("📌 你好啊，我来了，嘻嘻")

    for i, title in enumerate(st.session_state.generated_titles):
        title = title.strip()
        if title and not title.startswith("标题"):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f"**{i + 1}.** {title}")
            with col2:
                if st.button(f"选用", key=f"select_{i}"):
                    st.session_state.selected_title = title
                    st.rerun()

    # 显示选中的标题
    if st.session_state.selected_title:
        st.success(f"✅ 已选中：{st.session_state.selected_title}")

        # 生成正文大纲按钮
        if st.button("📄 正餐来了，有点诱人哦", type="primary"):
            with st.spinner("AI正在制作美食中..."):
                outline_prompt = f"""你是一个公众号推送编辑。根据以下标题和活动信息，生成推送正文大纲。
要求：
1. 包含：引言、正文要点（分3-4个部分）、结尾
2. 结构清晰，便于后续填充内容
3. 正文要点用简短的句子概括

标题：{st.session_state.selected_title}

活动信息：
{activity}

请直接输出大纲，不要加额外说明。"""

                outline = call_glm(outline_prompt)
                if "请求失败" not in outline:
                    st.session_state.generated_outline = outline
                else:
                    st.error(outline)

        # 显示大纲
        if st.session_state.generated_outline:
            st.markdown("---")
            st.subheader("📄 正餐来了，有点诱人哦")
            st.markdown(st.session_state.generated_outline)

            # 一键复制
            st.markdown("---")
            st.code(st.session_state.generated_outline, language="markdown")

            # 复制按钮（使用JavaScript）
            copy_html = f"""
            <button id="copyBtn" style="background-color:#4CAF50; color:white; padding:8px 16px; border:none; border-radius:4px; cursor:pointer;">
            📋 一键复制大纲
            </button>
            <script>
            document.getElementById('copyBtn').onclick = function() {{
                var text = {json.dumps(st.session_state.generated_outline)};
                navigator.clipboard.writeText(text);
                alert('已复制到剪贴板！');
            }};
            </script>
            """
            st.components.v1.html(copy_html, height=80)

# 如果没有输入就点生成
if generate_btn and not activity:
    st.warning("请先输入活动主题/内容")

# 页脚
st.markdown("---")
st.caption("推送排版助手 | 输入主题，AI帮你搞定标题和大纲")