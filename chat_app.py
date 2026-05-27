"""
市政给排水智能助手 - 最终优化版（修复思考过程重复、复制、LaTeX 指数问题）
- 按角色隔离对话历史
- 思考过程可折叠，最终结果清晰（彻底移除标签）
- 复制按钮复制干净答案（不含思考过程）
- LaTeX 公式正确渲染：m^2 -> m², m^3 -> m³ (通过规范化指数)
- 自动滚动
"""

import streamlit as st
import PyPDF2
import re
from AgentWithRAG import (
    query_standard,
    generate_calculation_report,
    review_bid_document,
    history_kb,
    local_kb
)

# ================= 页面配置 =================
st.set_page_config(
    page_title="市政给排水智能助手",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 初始化 session_state =================
if "role_messages" not in st.session_state:
    st.session_state.role_messages = {
        "规范查询": [],
        "给排水计算器": [],
        "招投标审核": []
    }
if "current_role" not in st.session_state:
    st.session_state.current_role = "规范查询"

# ================= 自定义 CSS =================
st.markdown("""
<style>
/* 全局字体基准 */
html, body, .stApp {
    font-size: 16px;
    font-family: 'Segoe UI', 'Roboto', '微软雅黑', sans-serif;
}
h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.3rem !important; }
h4, h5, h6, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 { font-size: 1.1rem !important; }
p, li, .stMarkdown, .stText, .stNumberInput label, .stSelectbox label {
    font-size: 1rem;
}
[data-testid="stSidebar"] {
    background-color: #eef2f6;
    padding-top: 1rem;
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stNumberInput {
    font-size: 0.9rem;
}
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}
.stButton > button {
    background-color: #2c7da0;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 0.4rem 1rem;
    font-weight: 500;
    transition: 0.2s;
}
.stButton > button:hover {
    background-color: #1f5068;
}
div[data-testid="stChatMessage"] {
    padding: 0.8rem;
    border-radius: 1rem;
    margin-bottom: 0.8rem;
}
div[data-testid="stChatMessage"][data-testid="user"] {
    background-color: #e8f0fe;
    border-top-right-radius: 0;
}
div[data-testid="stChatMessage"][data-testid="assistant"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-top-left-radius: 0;
}
/* 最终结果区域样式 */
.result-block {
    font-size: 1rem;
    line-height: 1.5;
}
/* 思考区域样式（用于 expander 内部） */
.thinking-block {
    background-color: #f1f3f5;
    border-left: 4px solid #adb5bd;
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #495057;
    font-style: italic;
}
</style>

<script>
function scrollToBottom() {
    const elements = document.querySelectorAll('[data-testid="stChatMessage"]');
    if(elements.length > 0) {
        elements[elements.length-1].scrollIntoView({ behavior: 'smooth' });
    }
}
window.addEventListener('load', scrollToBottom);
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)

# ================= LaTeX 增强修复函数 =================
def fix_latex(text: str) -> str:
    """
    1. 转义 & 符号（避免表格冲突）
    2. 规范化指数：将 m^3 转换为 m^{3}，m^2 -> m^{2} 等，确保上标正确渲染
    """
    # 转义 &
    text = re.sub(r'(?<!\\)&', r'\\&', text)
    
    # 规范化指数：匹配字母或右括号后跟 ^ 后跟数字（一个或多个）
    # 例如 m^3, m^2, L/s^2, (m)^2 等
    def _normalize_exp(match):
        base = match.group(1)   # 底数部分（字母或括号）
        exp = match.group(2)    # 指数数字
        return f"{base}^{{{exp}}}"
    
    # 匹配模式：([a-zA-Z\)])\^(\d+)
    text = re.sub(r'([a-zA-Z\)])\^(\d+)', _normalize_exp, text)
    
    return text

# ================= 辅助函数：彻底分离思考与答案 =================
def separate_think_and_answer(full_response: str):
    """
    从模型输出中提取所有 [THINK]...[/THINK] 块，并移除所有标签得到干净答案。
    返回 (think_content, clean_answer)
    """
    # 提取所有 [THINK]...[/THINK] 内容（非贪婪，支持跨行）
    think_pattern = r'\[THINK\](.*?)\[/THINK\]'
    think_blocks = re.findall(think_pattern, full_response, re.DOTALL)
    # 合并所有思考内容（防止模型输出多个块）
    think_content = "\n\n".join(block.strip() for block in think_blocks).strip()
    
    # 移除所有 [THINK]...[/THINK] 标签（包括标签本身）
    clean_answer = re.sub(think_pattern, '', full_response, flags=re.DOTALL).strip()
    # 清理多余空行（连续换行超过2个的替换为2个）
    clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)
    
    # 对最终答案再次应用 LaTeX 修复（确保指数规范化）
    clean_answer = fix_latex(clean_answer)
    
    # 如果思考内容为空但模型中确实有未闭合的 [THINK]（兜底）
    if not think_content and "[THINK]" in full_response:
        think_content = "（模型输出格式异常，请检查）"
    
    return think_content, clean_answer

def format_assistant_message(content: str):
    """解析并返回 (thinking_html, clean_answer)，用于显示和复制"""
    # 先对整个内容进行 LaTeX 修复（避免思考过程中也有未规范指数）
    content_fixed = fix_latex(content)
    think_content, clean_answer = separate_think_and_answer(content_fixed)
    
    if think_content:
        thinking_html = f'<div class="thinking-block">💭 <strong>思考过程</strong><br>{think_content.replace(chr(10), "<br>")}</div>'
    else:
        thinking_html = ""
    
    return thinking_html, clean_answer

def display_messages():
    current = st.session_state.current_role
    for idx, msg in enumerate(st.session_state.role_messages.get(current, [])):
        role = msg["role"]
        content = msg["content"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            if role == "user":
                st.markdown(content)
            else:
                think_html, clean_answer = format_assistant_message(content)
                if think_html:
                    with st.expander("💭 思考过程（点击展开/折叠）"):
                        st.markdown(think_html, unsafe_allow_html=True)
                if clean_answer:
                    st.markdown(clean_answer)
                    # 复制按钮：复制干净答案（不包含思考过程）
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        # 使用 st.button 触发 JavaScript 复制
                        # 注意：这里的复制实现依赖于 Streamlit 的 rerun 机制，为避免页面刷新，使用 st.toast 提示
                        if st.button("📋 复制答案", key=f"copy_{current}_{idx}", help="复制最终计算结果/报告"):
                            # 使用 st.markdown 注入 JS 复制剪贴板
                            js_code = f"""
                            <script>
                                (function() {{
                                    const text = {repr(clean_answer)};
                                    navigator.clipboard.writeText(text).then(function() {{
                                        console.log("复制成功");
                                    }}, function(err) {{
                                        console.error("复制失败", err);
                                    }});
                                }})();
                            </script>
                            """
                            st.markdown(js_code, unsafe_allow_html=True)
                            st.toast("已复制到剪贴板", icon="✅")

# ================= 侧边栏 =================
with st.sidebar:
    st.markdown('<div style="text-align: center; font-weight: bold; color: #2c7da0; margin-bottom: 1rem;">💧 市政给排水智能助手</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    new_role = st.radio("选择功能", ["规范查询", "给排水计算器", "招投标审核"], index=["规范查询", "给排水计算器", "招投标审核"].index(st.session_state.current_role))
    if new_role != st.session_state.current_role:
        st.session_state.current_role = new_role
        st.rerun()
    
    st.markdown("---")
    current_role = st.session_state.current_role
    
    if current_role == "给排水计算器":
        st.subheader("📝 计算参数")
        calc_type = st.selectbox("计算类型", [
            "管道流量", "沿程水头损失", "局部水头损失",
            "用水量计算", "水池有效容积", "水泵扬程估算"
        ])
        
        params = {}
        if calc_type == "管道流量":
            area = st.number_input("截面积 (m²)", 0.01, 10.0, 0.1, 0.01)
            velocity = st.number_input("流速 (m/s)", 0.1, 10.0, 1.0, 0.1)
            params = {"area": area, "velocity": velocity}
            calc_key = "flow"
            param_display = f"截面积 = {area} m², 流速 = {velocity} m/s"
        elif calc_type == "沿程水头损失":
            length = st.number_input("管长 (m)", 1.0, 10000.0, 100.0)
            velocity = st.number_input("流速 (m/s)", 0.1, 10.0, 1.0)
            diameter = st.number_input("管径 (m)", 0.01, 5.0, 0.2, 0.01)
            coeff = st.number_input("海曾-威廉系数 C", 80, 160, 120)
            params = {"length": length, "velocity": velocity, "diameter": diameter, "coeff": coeff}
            calc_key = "friction_loss"
            param_display = f"管长 = {length} m, 流速 = {velocity} m/s, 管径 = {diameter} m, C = {coeff}"
        elif calc_type == "局部水头损失":
            velocity = st.number_input("流速 (m/s)", 0.1, 10.0, 1.0)
            coeff = st.number_input("局部阻力系数 ζ", 0.01, 10.0, 0.5, 0.1)
            params = {"velocity": velocity, "coeff": coeff}
            calc_key = "local_loss"
            param_display = f"流速 = {velocity} m/s, 局部阻力系数 = {coeff}"
        elif calc_type == "用水量计算":
            unit_demand = st.number_input("人均用水定额 (L/人·d)", 50, 500, 200)
            population = st.number_input("服务人口", 1, 10000000, 1000)
            hours = st.number_input("用水时长 (h/d)", 1, 24, 24)
            params = {"unit_demand": unit_demand, "population": population, "hours": hours}
            calc_key = "water_demand"
            param_display = f"定额 = {unit_demand} L/人·d, 人口 = {population}, 时长 = {hours} h/d"
        elif calc_type == "水池有效容积":
            daily_demand = st.number_input("日用水量 (m³/d)", 1, 100000, 1000)
            reserve_hours = st.number_input("储备时间 (h)", 1, 72, 24)
            params = {"daily_demand": daily_demand, "reserve_hours": reserve_hours}
            calc_key = "tank_volume"
            param_display = f"日用水量 = {daily_demand} m³/d, 储备时间 = {reserve_hours} h"
        elif calc_type == "水泵扬程估算":
            static_head = st.number_input("静扬程 (m)", 0.0, 500.0, 20.0)
            friction_loss = st.number_input("沿程损失 (m)", 0.0, 100.0, 5.0)
            local_loss = st.number_input("局部损失 (m)", 0.0, 100.0, 2.0)
            outlet_pressure = st.number_input("出口压力水头 (m)", 0.0, 100.0, 10.0)
            params = {"static_head": static_head, "friction_loss": friction_loss, "local_loss": local_loss, "outlet_pressure": outlet_pressure}
            calc_key = "pump_head"
            param_display = f"静扬程 = {static_head} m, 沿程损失 = {friction_loss} m, 局部损失 = {local_loss} m, 出口压力水头 = {outlet_pressure} m"
        
        if st.button("🚀 开始计算", use_container_width=True):
            user_content = f"**计算类型**：{calc_type}\n**输入参数**：{param_display}"
            with st.spinner("正在计算并生成报告..."):
                report = generate_calculation_report(calc_key, params)
            st.session_state.role_messages[current_role].append({"role": "user", "content": user_content})
            st.session_state.role_messages[current_role].append({"role": "assistant", "content": report})
            st.rerun()
    
    elif current_role == "规范查询":
        st.info("💡 规范查询：将 PDF/TXT 放入 `knowledge/` 目录")
    elif current_role == "招投标审核":
        st.info("💡 招投标审核：可上传文件并启用历史学习")
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; font-size: 0.8rem; color: #6c757d;">© 2025 市政给排水智能助手<br>数据仅供设计参考</div>', unsafe_allow_html=True)

# ================= 主区域 =================
current_role = st.session_state.current_role

if current_role == "规范查询":
    st.header("📚 规范查询")
    st.markdown("在下方输入问题，系统将检索知识库并回答。")
    display_messages()
    question = st.chat_input("请输入您的问题...")
    if question:
        st.session_state.role_messages[current_role].append({"role": "user", "content": question})
        with st.spinner("检索中..."):
            answer = query_standard(question)
        st.session_state.role_messages[current_role].append({"role": "assistant", "content": answer})
        st.rerun()

elif current_role == "给排水计算器":
    st.header("🧮 给排水计算器")
    st.markdown("请在左侧边栏选择计算类型和参数，点击「开始计算」。")
    display_messages()

elif current_role == "招投标审核":
    st.header("📑 招投标文件合规审核（国标）")
    st.markdown("上传文件，系统将逐条审核。")
    
    with st.container():
        uploaded_file = st.file_uploader("选择文件", type=["pdf", "txt"], key="bid_upload")
        use_history = st.checkbox("🔍 启用历史案例学习", value=False)
        
        if st.button("🔍 开始审核", key="review_btn"):
            if uploaded_file is not None:
                file_content = ""
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        file_content += page.extract_text()
                else:
                    file_content = uploaded_file.read().decode("utf-8")
                if not file_content.strip():
                    st.error("文件内容为空")
                else:
                    user_req = f"**审核文件**：{uploaded_file.name}\n**启用历史学习**：{'是' if use_history else '否'}"
                    with st.spinner("审核中..."):
                        result = review_bid_document(file_content, use_history)
                    st.session_state.role_messages[current_role].append({"role": "user", "content": user_req})
                    st.session_state.role_messages[current_role].append({"role": "assistant", "content": result})
                    st.rerun()
    
    with st.expander("📚 历史案例管理（投喂学习用）"):
        hist_file = st.file_uploader("历史招投标文件", type=["pdf", "txt"], key="hist_upload")
        hist_opinion = st.text_area("人工审核意见")
        if st.button("开始学习"):
            if hist_file and hist_opinion.strip():
                content = ""
                if hist_file.type == "application/pdf":
                    reader = PyPDF2.PdfReader(hist_file)
                    for page in reader.pages:
                        content += page.extract_text()
                else:
                    content = hist_file.read().decode("utf-8")
                snippets = [p.strip() for p in content.split("\n\n") if p.strip()] or [content[:500]]
                opinions = [hist_opinion] * len(snippets)
                history_kb.add_document(hist_file.name, snippets, opinions)
                st.success(f"已学习 {hist_file.name}")
                st.rerun()
            else:
                st.error("请填写完整")
    
    display_messages()