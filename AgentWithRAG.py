"""
市政给排水智能助手 - 核心逻辑（语义检索 + TF-IDF 双引擎，增强健壮性）
包含：
- 文档分块（段落/句子）
- 语义检索（sentence-transformers） + TF-IDF fallback
- 规范名称优先匹配（GB/T 等）
- 提示词兜底（检索不足时模型主动提示）
- 向量索引缓存（pickle）
- 抑制因缺少 torchvision 产生的无用警告
- 修复模型加载失败时的降级逻辑，避免 NoneType 错误
- 兼容 Streamlit Cloud 和本地环境的环境变量/Secrets 读取
"""

import warnings
import os

# 抑制 transformers 库因缺少 torchvision 而抛出的警告和错误日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import re
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import PyPDF2
import dashscope
from dashscope import Generation

# ================= 配置区域（兼容本地环境变量和 Streamlit Secrets）=================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def get_api_key() -> str:
    """从环境变量或 Streamlit secrets 中获取 API Key"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get("DASHSCOPE_API_KEY")
            if api_key:
                return api_key
        except Exception:
            pass
    raise ValueError(
        "请在环境变量或 Streamlit secrets 中设置 DASHSCOPE_API_KEY。\n"
        "本地运行：`set DASHSCOPE_API_KEY=your_key` (Windows) 或 `export DASHSCOPE_API_KEY=your_key` (Linux/Mac)\n"
        "Streamlit Cloud：在 App Settings -> Secrets 中添加 DASHSCOPE_API_KEY。"
    )

DASHSCOPE_API_KEY = get_api_key()
dashscope.api_key = DASHSCOPE_API_KEY
MODEL_NAME = "qwen3.7-max"

# ================= 尝试导入语义检索库 =================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将使用 TF-IDF 检索。建议运行: pip install sentence-transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn 未安装，将使用简单词袋重叠检索（效果最差）。建议运行: pip install scikit-learn")

# ================= 本地知识库（语义检索 + TF-IDF 双引擎）=================
class LocalKnowledgeBase:
    def __init__(self, kb_dir="E:/hello-agents/code/chapter1/knowledge", cache_dir="E:/hello-agents/code/chapter1/cache"):
        self.kb_dir = kb_dir
        self.cache_dir = cache_dir
        self.documents = []          # 每个元素: {"filename": str, "page": int, "text": str, "norm_name": str}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None   # numpy array, shape (n_docs, dim)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_documents()
        
        # 构建检索索引（优先语义，fallback TF-IDF）
        if self.documents:
            self._build_or_load_index()

    def _get_cache_path(self, suffix: str) -> str:
        """根据知识库目录的哈希值生成缓存文件名"""
        kb_hash = hashlib.md5(self.kb_dir.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{kb_hash}_{suffix}.pkl")

    def _load_documents(self):
        """加载 PDF/TXT 文件，按段落分块，保留文件名和页码"""
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            return
        for filename in os.listdir(self.kb_dir):
            filepath = os.path.join(self.kb_dir, filename)
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                self._add_text_document(filename, text)
            elif filename.endswith(".pdf"):
                self._add_pdf_document(filename, filepath)

    def _add_text_document(self, filename, text):
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > 800:
                sentences = re.split(r'(?<=[。！？])', para)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        self.documents.append({
                            "filename": filename,
                            "page": 0,
                            "text": sent,
                            "norm_name": self._extract_norm_name(filename)
                        })
            else:
                self.documents.append({
                    "filename": filename,
                    "page": 0,
                    "text": para,
                    "norm_name": self._extract_norm_name(filename)
                })

    def _add_pdf_document(self, filename, filepath):
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text.strip():
                        continue
                    paragraphs = re.split(r'\n\s*\n', text)
                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                        if len(para) > 800:
                            sentences = re.split(r'(?<=[。！？])', para)
                            for sent in sentences:
                                sent = sent.strip()
                                if sent:
                                    self.documents.append({
                                        "filename": filename,
                                        "page": page_num + 1,
                                        "text": sent,
                                        "norm_name": self._extract_norm_name(filename)
                                    })
                        else:
                            self.documents.append({
                                "filename": filename,
                                "page": page_num + 1,
                                "text": para,
                                "norm_name": self._extract_norm_name(filename)
                            })
        except Exception as e:
            print(f"读取 PDF 失败 {filename}: {e}")

    def _extract_norm_name(self, filename: str) -> str:
        match = re.search(r'(GB/T?\s*\d{5,6}-\d{4})', filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    def _build_or_load_index(self):
        """优先加载缓存的语义索引，否则重新构建"""
        # 语义索引（sentence-transformers）
        if SENTENCE_TRANSFORMER_AVAILABLE:
            sem_cache = self._get_cache_path("semantic")
            if os.path.exists(sem_cache):
                try:
                    with open(sem_cache, "rb") as f:
                        cache_data = pickle.load(f)
                    self.semantic_embeddings = cache_data["embeddings"]
                    self.documents = cache_data["documents"]
                    print(f"从缓存加载语义索引，共 {len(self.documents)} 个文档块")
                except Exception as e:
                    print(f"语义索引缓存加载失败: {e}，将重新构建")
                    self._build_semantic_index()
                    # 保存缓存
                    with open(sem_cache, "wb") as f:
                        pickle.dump({
                            "embeddings": self.semantic_embeddings,
                            "documents": self.documents
                        }, f)
                    print("语义索引已缓存。")
            else:
                print("正在构建语义索引（首次启动较慢，后续将使用缓存）...")
                self._build_semantic_index()
                # 保存缓存
                with open(sem_cache, "wb") as f:
                    pickle.dump({
                        "embeddings": self.semantic_embeddings,
                        "documents": self.documents
                    }, f)
                print("语义索引已缓存。")
        # TF-IDF 索引（作为 fallback）
        if SKLEARN_AVAILABLE:
            self._build_tfidf_index()

    def _build_semantic_index(self):
        """使用 sentence-transformers 计算所有文档块的向量（本地加载）"""
        if not self.documents:
            return
        # 明确使用本地模型路径（您下载好的文件夹）
        local_model_path = r"E:\hello-agents\code\chapter1\paraphrase-multilingual-MiniLM-L12-v2"
        try:
            self.semantic_model = SentenceTransformer(local_model_path, local_files_only=True)
            texts = [doc["text"] for doc in self.documents]
            self.semantic_embeddings = self.semantic_model.encode(texts, show_progress_bar=True)
            print(f"语义模型加载成功，共 {len(self.documents)} 个文档块")
        except Exception as e:
            print(f"语义模型加载失败: {e}，将禁用语义检索，仅使用 TF-IDF。")
            self.semantic_model = None
            self.semantic_embeddings = None

    def _build_tfidf_index(self):
        if not self.documents:
            return
        texts = [doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.15) -> List[Dict]:
        """
        检索相关文档块：
        1. 若 query 中包含规范编号（如 GB 50013-2018），则只检索该规范的块
        2. 优先使用语义检索（余弦相似度），若不可用则降级到 TF-IDF
        3. 过滤掉相似度低于 threshold 的结果
        """
        if not self.documents:
            return []

        # 规范过滤
        target_norm = None
        norm_pattern = r'(GB/T?\s*\d{5,6}-\d{4})'
        match = re.search(norm_pattern, query, re.IGNORECASE)
        if match:
            target_norm = match.group(1).upper()
            docs = [doc for doc in self.documents if target_norm in doc["norm_name"]]
            if not docs:
                print(f"警告: 指定规范 {target_norm} 但未找到对应文档，使用全量检索")
                docs = self.documents
        else:
            docs = self.documents

        if not docs:
            return []

        # 语义检索（优先）
        if (SENTENCE_TRANSFORMER_AVAILABLE and self.semantic_embeddings is not None 
                and self.semantic_model is not None):
            if len(docs) == len(self.documents):
                # 全量检索
                query_vec = self.semantic_model.encode([query])
                similarities = cosine_similarity(query_vec, self.semantic_embeddings).flatten()
                indices = similarities.argsort()[-top_k:][::-1]
                results = []
                for idx in indices:
                    if similarities[idx] >= similarity_threshold:
                        results.append(self.documents[idx])
                return results
            else:
                # 子集检索：临时编码子集文本
                sub_texts = [doc["text"] for doc in docs]
                sub_embeddings = self.semantic_model.encode(sub_texts)
                query_vec = self.semantic_model.encode([query])
                similarities = cosine_similarity(query_vec, sub_embeddings).flatten()
                indices = similarities.argsort()[-top_k:][::-1]
                results = []
                for idx in indices:
                    if similarities[idx] >= similarity_threshold:
                        results.append(docs[idx])
                return results

        # Fallback: TF-IDF 检索
        if SKLEARN_AVAILABLE and self.vectorizer is not None:
            if len(docs) == len(self.documents):
                query_vec = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                indices = similarities.argsort()[-top_k:][::-1]
                results = []
                for idx in indices:
                    if similarities[idx] >= similarity_threshold:
                        results.append(self.documents[idx])
                return results
            else:
                sub_texts = [doc["text"] for doc in docs]
                sub_vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'(?u)\b\w+\b')
                sub_matrix = sub_vectorizer.fit_transform(sub_texts)
                query_vec = sub_vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, sub_matrix).flatten()
                indices = similarities.argsort()[-top_k:][::-1]
                results = []
                for idx in indices:
                    if similarities[idx] >= similarity_threshold:
                        results.append(docs[idx])
                return results

        # 最终 fallback：简单词袋重叠
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
            doc_words = set(doc["text"].lower().split())
            common = query_words.intersection(doc_words)
            score = len(common) / max(len(query_words), 1)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

local_kb = LocalKnowledgeBase()

# ================= 给排水计算工具 =================
def calculate_flow(area_m2: float, velocity_m_s: float) -> Dict[str, float]:
    flow_m3_s = area_m2 * velocity_m_s
    flow_l_s = flow_m3_s * 1000
    return {"flow_m3_s": round(flow_m3_s, 3), "flow_l_s": round(flow_l_s, 1)}

def calculate_friction_loss(length_m: float, velocity_m_s: float, diameter_m: float, coefficient: float = 120) -> float:
    area = 3.14159 * (diameter_m/2)**2
    flow = area * velocity_m_s
    hf = (10.67 * length_m * (flow**1.852)) / (coefficient**1.852 * (diameter_m**4.87))
    return round(hf, 2)

def calculate_local_loss(velocity_m_s: float, local_coefficient: float) -> float:
    g = 9.81
    hm = local_coefficient * (velocity_m_s**2) / (2 * g)
    return round(hm, 2)

def calculate_water_demand(unit_demand: float, population: int, usage_hours: float = 24) -> Dict[str, float]:
    avg_hour_m3 = unit_demand * population / 1000 / usage_hours
    max_hour_m3 = avg_hour_m3 * 1.3
    return {"avg_hour_m3": round(avg_hour_m3, 2), "max_hour_m3": round(max_hour_m3, 2)}

def calculate_tank_volume(daily_demand_m3: float, reserve_hours: float = 24) -> float:
    volume = daily_demand_m3 * reserve_hours / 24
    return round(volume, 1)

def calculate_pump_head(static_head_m: float, friction_loss_m: float, local_loss_m: float, outlet_pressure_m: float = 10) -> float:
    return round(static_head_m + friction_loss_m + local_loss_m + outlet_pressure_m, 2)

# ================= 招投标审核（国标规则 + 历史学习）=================
BID_REVIEW_RULES = [
    {"id": "ZTB-01", "content": "招标项目应当依法发布招标公告，且公告内容完整、合规。", "risk": "高"},
    {"id": "ZTB-02", "content": "不得设置不合理的资质要求或特定区域、特定行业的业绩门槛，排斥潜在投标人。", "risk": "高"},
    {"id": "ZTB-03", "content": "投标人之间存在控股、管理关系的，不得参加同一标段投标。", "risk": "高"},
    {"id": "ZTB-04", "content": "评标办法应当明确，不得含有倾向性或歧视性条款。", "risk": "中"},
    {"id": "ZTB-05", "content": "中标候选人公示应当载明中标候选人排序、名称、投标报价、质量、工期等信息。", "risk": "中"},
    {"id": "ZTB-06", "content": "合同文本应当与招标文件、中标人的投标文件实质性内容一致。", "risk": "高"},
    {"id": "GPS-01", "content": "给排水管道材料应符合 GB/T 13663 或相应国标要求，并明确规格型号。", "risk": "高"},
    {"id": "GPS-02", "content": "水压试验、闭水试验等验收标准应引用 GB 50268 等现行国标。", "risk": "中"},
]

from bid_history_kb import HistoricalBidKB
history_kb = HistoricalBidKB()

def review_bid_document(content: str, use_history: bool = False) -> str:
    history_refs = []
    if use_history:
        query = content[:2000] if len(content) > 2000 else content
        history_refs = history_kb.search(query, top_k=3)
    
    rules_text = "\n".join([f"- **{r['id']}**：{r['content']}（风险等级：{r['risk']}）" for r in BID_REVIEW_RULES])
    history_text = ""
    if use_history and history_refs:
        history_text = "以下是与当前文件相似的历史招投标案例及审核意见，可作参考：\n"
        for ref in history_refs:
            history_text += f"- **文件**：{ref['filename']}（相似度：{ref['score']}）\n  **相关内容**：{ref['snippet']}\n  **历史审核意见**：{ref['opinion']}\n"
    else:
        history_text = "未启用历史案例学习，或暂无相似案例。"
    
    system_msg = """你是一位资深的招投标合规专家。必须严格遵守以下输出格式：
1. 整个回复必须以 `[THINK]` 开头，紧接着是你的分析思考过程，然后以 `[/THINK]` 结束思考部分。
2. 思考过程结束后，立即开始输出最终的审核报告。
3. **最终审核报告中禁止再出现 `[THINK]` 或 `[/THINK]` 标签**，也禁止重复思考过程中的任何句子。
4. 思考内容只出现在 `[THINK]...[/THINK]` 块内，最终报告独立且干净。
不遵守格式将导致前端解析失败。"""
    
    user_prompt = f"""请根据以下国标条款对用户提供的招投标文件进行逐条审核。

一、国标审核规则库
{rules_text}

二、待审核文件内容
{content}

三、历史案例参考
{history_text}

四、任务要求
输出结构化的审核报告，请使用以下公文格式（不要使用 Markdown 标题符号 `#`，改用中文序号）：
一、总体评价（合规/基本合规/存在风险）
二、逐条审核结果（每条国标是否满足，若不满足，指出文件中的具体位置和内容，并给出修改建议）
三、高风险项汇总（列出所有风险等级为“高”的不符合项）
四、可借鉴的历史经验（若启用了历史案例，总结与本文件相关的经验）

注意：先输出思考过程（用 `[THINK]...[/THINK]` 包裹），然后输出报告，报告内不要重复思考内容。
"""
    return call_llm(user_prompt, system_message=system_msg)

# ================= 通用大模型调用 =================
def call_llm(prompt: str, system_message: str = None) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    try:
        response = Generation.call(model=MODEL_NAME, messages=messages, result_format='message')
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"API 调用失败: {response.code} - {response.message}"
    except Exception as e:
        return f"请求异常: {str(e)}"

# ================= 规范查询（增强版：提示词兜底 + 公文格式）=================
def query_standard(question: str) -> str:
    docs = local_kb.search(question, top_k=8, similarity_threshold=0.15)
    
    low_recall = len(docs) < 2
    context = ""
    for doc in docs:
        source = f"{doc['filename']}"
        if doc.get('page'):
            source += f" 第{doc['page']}页"
        context += f"\n【来源：{source}】\n{doc['text']}\n"
    
    system_msg = """你是市政给排水规范查询助手。必须严格遵守以下输出格式：
1. 整个回复必须以 `[THINK]` 开头，紧接着是你的思考过程，然后以 `[/THINK]` 结束思考部分。
2. 思考过程结束后，立即开始输出最终答案（规范条文引用和建议）。
3. **最终答案中禁止再出现 `[THINK]` 或 `[/THINK]` 标签**，也禁止重复思考过程中的任何句子。
4. 引用具体条文时必须注明文件名和页码。
如果检索到的条文不足以回答用户问题，请如实说明，并建议用户提供更具体的规范编号或关键词。"""
    
    if low_recall or not context.strip():
        system_msg += "\n【特别提醒】当前知识库中与用户问题高度相关的规范条文较少。如果你认为无法基于已有条文给出充分回答，请明确告知用户，并建议用户：1) 提供更具体的规范编号（如 GB 50013-2018）；2) 更换关键词；3) 确认规范文件是否已放入 knowledge 目录。"
    
    user_prompt = f"""请严格根据以下规范条文内容回答用户的问题。

规范条文：
{context if context else "（未检索到相关规范条文）"}

用户问题：{question}

要求：
- 引用具体条文（注明文件名和页码）。
- 如果问题超出条文范围，请如实说明，但不要编造规范不存在的信息。
- 先输出思考过程（用 `[THINK]...[/THINK]` 包裹），然后输出答案。
- **答案部分请使用以下公文格式**（不要使用 Markdown 标题 `#`，改用中文序号）：
  一、问题概述
  二、相关规范条文（逐条列出，每条注明来源）
  三、结论与建议
- 段落之间空一行。

请现在输出。
"""
    return call_llm(user_prompt, system_message=system_msg)

# ================= 给排水计算器（公文格式）=================
def generate_calculation_report(calc_type: str, params: dict) -> str:
    # 计算
    result = None
    if calc_type == "flow":
        result = calculate_flow(params['area'], params['velocity'])
    elif calc_type == "friction_loss":
        result = calculate_friction_loss(params['length'], params['velocity'], params['diameter'], params.get('coeff', 120))
    elif calc_type == "local_loss":
        result = calculate_local_loss(params['velocity'], params['coeff'])
    elif calc_type == "water_demand":
        result = calculate_water_demand(params['unit_demand'], params['population'], params.get('hours', 24))
    elif calc_type == "tank_volume":
        result = calculate_tank_volume(params['daily_demand'], params.get('reserve_hours', 24))
    elif calc_type == "pump_head":
        result = calculate_pump_head(params['static_head'], params['friction_loss'], params['local_loss'], params.get('outlet_pressure', 10))
    else:
        return "不支持的计算类型"
    
    system_msg = """你是一个给排水专业计算器。必须严格遵守以下输出格式：
1. 整个回复必须以 `[THINK]` 开头，紧接着是你的思考过程，然后以 `[/THINK]` 结束思考部分。
2. 思考过程结束后，立即开始输出最终的计算书正文。
3. **最终计算书正文中禁止再出现 `[THINK]` 或 `[/THINK]` 标签**，也禁止重复思考过程中的任何句子。
4. 思考内容只出现在 `[THINK]...[/THINK]` 块内，计算书正文独立且干净。
不遵守格式将导致前端解析失败。"""
    
    user_prompt = f"""请生成一份给排水设计计算书。

计算类型: {calc_type}
输入参数: {params}
计算结果: {result}

要求：
1. **先输出思考过程**：以 `[THINK]` 开头，`[/THINK]` 结尾。内容只包含：计算依据、公式选择、单位换算、参数代入步骤、中间数值。**严禁**在思考过程中出现“计算书格式要求”等指令性文字。
2. **然后输出最终计算书**：紧接在 `[/THINK]` 之后输出，必须严格使用以下**公文格式**（不要使用 Markdown 标题符号 `#`，改用中文序号）：
   一、计算项目
   二、计算依据
   三、已知参数（建议用表格）
   四、计算公式（使用 LaTeX，块级公式用 $$...$$，行内公式用 $...$）
   五、计算过程
   六、计算结果（建议用表格）
   七、结论
3. **严禁**：在 `[/THINK]` 之后再次出现 `[THINK]` 或任何思考过程内容。最终计算书内不得重复思考过程。

请现在输出。
"""
    return call_llm(user_prompt, system_message=system_msg)