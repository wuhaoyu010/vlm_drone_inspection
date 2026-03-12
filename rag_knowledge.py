"""
RAG知识库模块 - 为高速公路检测任务提供专业知识检索
支持法律法规、安全规范、养护标准等知识库
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# 延迟导入，避免未安装时报错
def _import_langchain():
    """延迟导入langchain组件"""
    global FAISS, HuggingFaceBgeEmbeddings, RecursiveCharacterTextSplitter
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return True
    except ImportError:
        return False

LANGCHAIN_AVAILABLE = _import_langchain()


class KnowledgeBase:
    """知识库管理器 - 支持PDF文档、文本文件的知识检索

    用于增强L2分析过程和L3风险评估的专业性
    """

    _instance = None  # 单例模式

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        knowledge_dir: str = "./knowledge_base",
        embedding_model_path: str = None,
        cache_dir: str = "./rag_cache",
        use_rag: bool = True
    ):
        """初始化知识库

        Args:
            knowledge_dir: 知识文档目录（PDF、TXT文件）
            embedding_model_path: 本地嵌入模型路径
            cache_dir: 向量库缓存目录
            use_rag: 是否启用RAG检索
        """
        if self._initialized:
            return

        self.knowledge_dir = knowledge_dir
        self.cache_dir = cache_dir
        self.use_rag = use_rag and LANGCHAIN_AVAILABLE

        # 嵌入模型路径（优先使用本地模型）
        self.embedding_model_path = embedding_model_path or os.environ.get(
            "EMBEDDING_MODEL_PATH",
            "./bge-small-zh-v1.5"
        )

        self.vectorstore = None
        self.retriever = None
        self.embedding_model = None

        if self.use_rag:
            self._init_knowledge_base()

        self._initialized = True

    def _init_knowledge_base(self):
        """初始化知识库向量存储"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)

        # 检查嵌入模型是否存在
        if not os.path.exists(self.embedding_model_path):
            print(f"[RAG] 嵌入模型不存在: {self.embedding_model_path}")
            print(f"[RAG] 请下载BGE模型或设置环境变量 EMBEDDING_MODEL_PATH")
            self.use_rag = False
            return

        try:
            # 加载嵌入模型
            print(f"[RAG] 加载嵌入模型: {self.embedding_model_path}")
            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # 检查缓存
            cache_path = os.path.join(self.cache_dir, "faiss_index")
            fingerprint_path = os.path.join(self.cache_dir, "fingerprint.txt")

            current_fingerprint = self._get_knowledge_fingerprint()

            if os.path.exists(cache_path) and os.path.exists(fingerprint_path):
                with open(fingerprint_path, 'r', encoding='utf-8') as f:
                    saved_fingerprint = f.read().strip()

                if current_fingerprint == saved_fingerprint:
                    print("[RAG] 从缓存加载向量库...")
                    self.vectorstore = FAISS.load_local(
                        cache_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    self.retriever = self.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    print(f"[RAG] 知识库加载完成")
                    return

            # 构建新的向量库
            print("[RAG] 构建知识库向量索引...")
            documents = self._load_documents()

            if documents:
                # 分块
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", "。", "；", " ", ""]
                )
                splits = text_splitter.split_documents(documents)
                print(f"[RAG] 生成 {len(splits)} 个文本块")

                # 构建向量库
                self.vectorstore = FAISS.from_documents(splits, self.embedding_model)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

                # 保存缓存
                self.vectorstore.save_local(cache_path)
                with open(fingerprint_path, 'w', encoding='utf-8') as f:
                    f.write(current_fingerprint)

                print(f"[RAG] 知识库构建完成，共 {len(splits)} 个知识片段")
            else:
                print(f"[RAG] 知识目录为空，请添加PDF或TXT文件到: {self.knowledge_dir}")
                self.use_rag = False

        except Exception as e:
            print(f"[RAG] 初始化失败: {e}")
            self.use_rag = False

    def _get_knowledge_fingerprint(self) -> str:
        """生成知识库目录指纹"""
        if not os.path.exists(self.knowledge_dir):
            return ""

        files = sorted([
            f for f in os.listdir(self.knowledge_dir)
            if f.endswith(('.pdf', '.txt', '.md'))
        ])

        if not files:
            return ""

        fingerprint_lines = []
        for f in files:
            path = os.path.join(self.knowledge_dir, f)
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
            fingerprint_lines.append(f"{f}:{mtime}:{size}")

        return "\n".join(fingerprint_lines)

    def _load_documents(self) -> List:
        """加载知识库文档"""
        documents = []

        if not os.path.exists(self.knowledge_dir):
            return documents

        # 加载PDF文件
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader

            for filename in os.listdir(self.knowledge_dir):
                filepath = os.path.join(self.knowledge_dir, filename)

                if filename.endswith('.pdf'):
                    print(f"[RAG] 加载PDF: {filename}")
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)

                elif filename.endswith(('.txt', '.md')):
                    print(f"[RAG] 加载文本: {filename}")
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)

        except Exception as e:
            print(f"[RAG] 文档加载失败: {e}")

        return documents

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关知识

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关知识片段列表
        """
        if not self.use_rag or self.retriever is None:
            return []

        try:
            docs = self.retriever.get_relevant_documents(query)
            results = []
            for doc in docs[:top_k]:
                source = doc.metadata.get('source', '未知')
                page = doc.metadata.get('page', '')
                content = doc.page_content.strip()
                results.append(f"[来源: {source}{' 第'+str(page)+'页' if page else ''}]\n{content}")
            return results
        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return []

    def retrieve_for_task2(
        self,
        vehicle_type: str,
        parking_location: str,
        violation_type: str = "违停"
    ) -> Dict[str, str]:
        """为Task2检索相关知识

        Args:
            vehicle_type: 车辆类型
            parking_location: 停车位置
            violation_type: 违规类型

        Returns:
            包含法律法规、安全规范、处置建议的字典
        """
        if not self.use_rag:
            return self._get_default_knowledge(vehicle_type, parking_location)

        result = {
            "laws_regulations": "",
            "safety_standards": "",
            "disposal_guidance": ""
        }

        # 检索法律法规
        law_query = f"高速公路{parking_location}违停 法律法规 处罚"
        law_docs = self.retrieve(law_query, top_k=2)
        result["laws_regulations"] = "\n".join(law_docs) if law_docs else ""

        # 检索安全规范
        safety_query = f"{parking_location}停车 安全隐患 风险评估"
        safety_docs = self.retrieve(safety_query, top_k=2)
        result["safety_standards"] = "\n".join(safety_docs) if safety_docs else ""

        # 检索处置建议
        disposal_query = f"违停车辆 处置措施 养护建议 {vehicle_type}"
        disposal_docs = self.retrieve(disposal_query, top_k=2)
        result["disposal_guidance"] = "\n".join(disposal_docs) if disposal_docs else ""

        return result

    def _get_default_knowledge(self, vehicle_type: str, parking_location: str) -> Dict[str, str]:
        """默认知识（RAG未启用时使用）"""
        return {
            "laws_regulations": "根据《道路交通安全法》，高速公路违停属于违法行为，可处以罚款200元、记9分的处罚。",
            "safety_standards": "高速公路违停车辆对后方来车构成安全隐患，可能引发追尾事故。",
            "disposal_guidance": "建议通知路政人员前往现场处理，引导车辆驶离或采取拖移措施。"
        }

    def is_available(self) -> bool:
        """检查知识库是否可用"""
        return self.use_rag and self.retriever is not None


# 全局知识库实例
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(
    knowledge_dir: str = None,
    embedding_model_path: str = None,
    use_rag: bool = True
) -> KnowledgeBase:
    """获取知识库单例实例

    Args:
        knowledge_dir: 知识文档目录
        embedding_model_path: 嵌入模型路径
        use_rag: 是否启用RAG

    Returns:
        KnowledgeBase实例
    """
    global _knowledge_base

    if _knowledge_base is None:
        # 使用默认路径
        if knowledge_dir is None:
            knowledge_dir = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        if embedding_model_path is None:
            embedding_model_path = os.environ.get("EMBEDDING_MODEL_PATH", "./bge-small-zh-v1.5")

        _knowledge_base = KnowledgeBase(
            knowledge_dir=knowledge_dir,
            embedding_model_path=embedding_model_path,
            use_rag=use_rag
        )

    return _knowledge_base


def retrieve_knowledge_for_l2(
    vehicle_type: str,
    parking_location: str,
    context: str = ""
) -> str:
    """为L2分析过程检索相关知识

    Args:
        vehicle_type: 车辆类型
        parking_location: 停车位置
        context: 额外上下文

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return ""

    knowledge = kb.retrieve_for_task2(vehicle_type, parking_location)

    formatted = []
    if knowledge.get("laws_regulations"):
        formatted.append(f"【相关法规】{knowledge['laws_regulations']}")
    if knowledge.get("safety_standards"):
        formatted.append(f"【安全规范】{knowledge['safety_standards']}")

    return "\n".join(formatted)


# ==================== Task1: 抛洒物检测知识检索 ====================

def retrieve_knowledge_for_task1_l2(
    debris_type: str,
    location: str,
    size: str = "未知"
) -> str:
    """为Task1 L2分析过程检索相关知识

    Args:
        debris_type: 抛洒物类型（如纸箱、轮胎、建筑材料等）
        location: 抛洒物位置（行车道、应急车道等）
        size: 抛洒物大小

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task1(debris_type, location)

    # 构建检索查询
    queries = [
        f"高速公路抛洒物 {debris_type} 安全风险",
        f"{location}抛洒物 处置措施",
        f"路面障碍物 {debris_type} 养护规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task1(debris_type, location)


def retrieve_knowledge_for_task1_l3(
    debris_type: str,
    location: str,
    risk_level: str
) -> str:
    """为Task1 L3风险评估检索相关知识

    Args:
        debris_type: 抛洒物类型
        location: 抛洒物位置
        risk_level: 风险等级

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task1(debris_type, location)

    # 构建检索查询
    queries = [
        f"抛洒物清理 处置流程 {location}",
        f"{debris_type} 养护处置建议",
        f"路面障碍物清除规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task1(debris_type, location)


def _get_default_knowledge_task1(debris_type: str, location: str) -> str:
    """Task1默认知识（RAG未启用时使用）"""
    return f"根据《公路养护技术规范》，{location}抛洒物属于路面障碍物，影响行车安全。{debris_type}类抛洒物可能造成车辆避让不及引发交通事故。"


def _get_default_disposal_task1(debris_type: str, location: str) -> str:
    """Task1默认处置建议"""
    return f"建议及时清理{location}的{debris_type}，设置警示标志提醒过往车辆，必要时封闭车道进行作业。"


# ==================== Task3: 路面病害检测知识检索 ====================

def retrieve_knowledge_for_task3_l2(
    disease_type: str,
    severity: str = "未知",
    location: str = ""
) -> str:
    """为Task3 L2分析过程检索相关知识

    Args:
        disease_type: 病害类型（裂缝、坑洼、积水等）
        severity: 严重程度
        location: 病害位置

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task3(disease_type, severity)

    # 构建检索查询
    queries = [
        f"高速公路路面{disease_type} 病害成因 分析",
        f"路面{disease_type} 检测标准 评定",
        f"{disease_type}病害 {'严重' if severity in ['严重', '重度'] else '轻微'} 处理"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task3(disease_type, severity)


def retrieve_knowledge_for_task3_l3(
    disease_type: str,
    severity: str,
    area: str = ""
) -> str:
    """为Task3 L3风险评估检索相关知识

    Args:
        disease_type: 病害类型
        severity: 严重程度
        area: 病害面积/范围

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task3(disease_type, severity)

    # 构建检索查询
    queries = [
        f"路面{disease_type} 养护维修方案",
        f"{disease_type}病害 {'紧急' if severity in ['严重', '重度'] else '常规'} 修补",
        f"高速公路养护 {disease_type} 处置规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task3(disease_type, severity)


def _get_default_knowledge_task3(disease_type: str, severity: str) -> str:
    """Task3默认知识"""
    severity_map = {
        "轻微": "轻度",
        "中度": "中度",
        "严重": "重度",
        "重度": "重度"
    }
    level = severity_map.get(severity, "")
    return f"根据《公路技术状况评定标准》，路面{disease_type}属于{level}病害，影响路面平整度和行车舒适性。"


def _get_default_disposal_task3(disease_type: str, severity: str) -> str:
    """Task3默认处置建议"""
    if severity in ["严重", "重度"]:
        return f"建议尽快对路面{disease_type}进行修复，采用专业养护材料和技术方案，确保修复质量。"
    return f"建议适时安排路面{disease_type}的养护维修，防止病害进一步扩展。"


# ==================== Task4: 路外病害检测知识检索 ====================

def retrieve_knowledge_for_task4_l2(
    disease_type: str,
    location: str = ""
) -> str:
    """为Task4 L2分析过程检索相关知识

    Args:
        disease_type: 病害类型（边坡滑坡、排水沟积水、排水沟破损等）
        location: 病害位置

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task4(disease_type, location)

    # 构建检索查询
    queries = [
        f"高速公路{disease_type} 病害分析 成因",
        f"{disease_type} 安全风险评估",
        f"路外设施 {disease_type} 养护检查"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task4(disease_type, location)


def retrieve_knowledge_for_task4_l3(
    disease_type: str,
    severity: str,
    impact: str = ""
) -> str:
    """为Task4 L3风险评估检索相关知识

    Args:
        disease_type: 病害类型
        severity: 严重程度
        impact: 影响范围

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task4(disease_type, severity)

    # 构建检索查询
    queries = [
        f"{disease_type} 防治措施 养护方案",
        f"高速公路 {disease_type} 应急处置",
        f"{disease_type} 修复工程 规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task4(disease_type, severity)


def _get_default_knowledge_task4(disease_type: str, location: str) -> str:
    """Task4默认知识"""
    knowledge_map = {
        "边坡滑坡": "边坡滑坡可能导致土石方坍塌，危及道路安全和行车安全。",
        "排水沟积水": "排水沟积水影响排水功能，可能导致路基软化。",
        "排水沟破损": "排水沟破损影响排水效率，可能加剧路基病害。"
    }
    return knowledge_map.get(disease_type, f"{disease_type}可能影响高速公路运营安全。")


def _get_default_disposal_task4(disease_type: str, severity: str) -> str:
    """Task4默认处置建议"""
    disposal_map = {
        "边坡滑坡": "建议立即采取边坡防护措施，必要时封闭交通，组织专业队伍进行治理。",
        "排水沟积水": "建议及时疏通排水沟，清理淤积物，恢复排水功能。",
        "排水沟破损": "建议修复排水沟破损部位，确保排水系统正常运行。"
    }
    return disposal_map.get(disease_type, f"建议及时处置{disease_type}，确保高速公路安全运营。")


def retrieve_knowledge_for_l3(
    vehicle_type: str,
    parking_location: str,
    risk_level: str
) -> str:
    """为L3风险评估检索相关知识

    Args:
        vehicle_type: 车辆类型
        parking_location: 停车位置
        risk_level: 风险等级

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return ""

    knowledge = kb.retrieve_for_task2(vehicle_type, parking_location)

    formatted = []
    if knowledge.get("disposal_guidance"):
        formatted.append(f"【处置依据】{knowledge['disposal_guidance']}")

    return "\n".join(formatted)


# ==================== Task1: 抛洒物检测知识检索 ====================

def retrieve_knowledge_for_task1_l2(
    debris_type: str,
    location: str,
    size: str = "未知"
) -> str:
    """为Task1 L2分析过程检索相关知识

    Args:
        debris_type: 抛洒物类型（如纸箱、轮胎、建筑材料等）
        location: 抛洒物位置（行车道、应急车道等）
        size: 抛洒物大小

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task1(debris_type, location)

    # 构建检索查询
    queries = [
        f"高速公路抛洒物 {debris_type} 安全风险",
        f"{location}抛洒物 处置措施",
        f"路面障碍物 {debris_type} 养护规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task1(debris_type, location)


def retrieve_knowledge_for_task1_l3(
    debris_type: str,
    location: str,
    risk_level: str
) -> str:
    """为Task1 L3风险评估检索相关知识

    Args:
        debris_type: 抛洒物类型
        location: 抛洒物位置
        risk_level: 风险等级

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task1(debris_type, location)

    # 构建检索查询
    queries = [
        f"抛洒物清理 处置流程 {location}",
        f"{debris_type} 养护处置建议",
        f"路面障碍物清除规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task1(debris_type, location)


def _get_default_knowledge_task1(debris_type: str, location: str) -> str:
    """Task1默认知识（RAG未启用时使用）"""
    return f"根据《公路养护技术规范》，{location}抛洒物属于路面障碍物，影响行车安全。{debris_type}类抛洒物可能造成车辆避让不及引发交通事故。"


def _get_default_disposal_task1(debris_type: str, location: str) -> str:
    """Task1默认处置建议"""
    return f"建议及时清理{location}的{debris_type}，设置警示标志提醒过往车辆，必要时封闭车道进行作业。"


# ==================== Task3: 路面病害检测知识检索 ====================

def retrieve_knowledge_for_task3_l2(
    disease_type: str,
    severity: str = "未知",
    location: str = ""
) -> str:
    """为Task3 L2分析过程检索相关知识

    Args:
        disease_type: 病害类型（裂缝、坑洼、积水等）
        severity: 严重程度
        location: 病害位置

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task3(disease_type, severity)

    # 构建检索查询
    queries = [
        f"高速公路路面{disease_type} 病害成因 分析",
        f"路面{disease_type} 检测标准 评定",
        f"{disease_type}病害 {'严重' if severity in ['严重', '重度'] else '轻微'} 处理"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task3(disease_type, severity)


def retrieve_knowledge_for_task3_l3(
    disease_type: str,
    severity: str,
    area: str = ""
) -> str:
    """为Task3 L3风险评估检索相关知识

    Args:
        disease_type: 病害类型
        severity: 严重程度
        area: 病害面积/范围

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task3(disease_type, severity)

    # 构建检索查询
    queries = [
        f"路面{disease_type} 养护维修方案",
        f"{disease_type}病害 {'紧急' if severity in ['严重', '重度'] else '常规'} 修补",
        f"高速公路养护 {disease_type} 处置规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task3(disease_type, severity)


def _get_default_knowledge_task3(disease_type: str, severity: str) -> str:
    """Task3默认知识"""
    severity_map = {
        "轻微": "轻度",
        "中度": "中度",
        "严重": "重度",
        "重度": "重度"
    }
    level = severity_map.get(severity, "")
    return f"根据《公路技术状况评定标准》，路面{disease_type}属于{level}病害，影响路面平整度和行车舒适性。"


def _get_default_disposal_task3(disease_type: str, severity: str) -> str:
    """Task3默认处置建议"""
    if severity in ["严重", "重度"]:
        return f"建议尽快对路面{disease_type}进行修复，采用专业养护材料和技术方案，确保修复质量。"
    return f"建议适时安排路面{disease_type}的养护维修，防止病害进一步扩展。"


# ==================== Task4: 路外病害检测知识检索 ====================

def retrieve_knowledge_for_task4_l2(
    disease_type: str,
    location: str = ""
) -> str:
    """为Task4 L2分析过程检索相关知识

    Args:
        disease_type: 病害类型（边坡滑坡、排水沟积水、排水沟破损等）
        location: 病害位置

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_knowledge_task4(disease_type, location)

    # 构建检索查询
    queries = [
        f"高速公路{disease_type} 病害分析 成因",
        f"{disease_type} 安全风险评估",
        f"路外设施 {disease_type} 养护检查"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=2)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:3])

    return _get_default_knowledge_task4(disease_type, location)


def retrieve_knowledge_for_task4_l3(
    disease_type: str,
    severity: str,
    impact: str = ""
) -> str:
    """为Task4 L3风险评估检索相关知识

    Args:
        disease_type: 病害类型
        severity: 严重程度
        impact: 影响范围

    Returns:
        格式化的知识内容
    """
    kb = get_knowledge_base()

    if not kb.is_available():
        return _get_default_disposal_task4(disease_type, severity)

    # 构建检索查询
    queries = [
        f"{disease_type} 防治措施 养护方案",
        f"高速公路 {disease_type} 应急处置",
        f"{disease_type} 修复工程 规范"
    ]

    results = []
    for query in queries:
        docs = kb.retrieve(query, top_k=1)
        if docs:
            results.extend(docs)

    if results:
        return "\n".join(results[:2])

    return _get_default_disposal_task4(disease_type, severity)


def _get_default_knowledge_task4(disease_type: str, location: str) -> str:
    """Task4默认知识"""
    knowledge_map = {
        "边坡滑坡": "边坡滑坡可能导致土石方坍塌，危及道路安全和行车安全。",
        "排水沟积水": "排水沟积水影响排水功能，可能导致路基软化。",
        "排水沟破损": "排水沟破损影响排水效率，可能加剧路基病害。"
    }
    return knowledge_map.get(disease_type, f"{disease_type}可能影响高速公路运营安全。")


def _get_default_disposal_task4(disease_type: str, severity: str) -> str:
    """Task4默认处置建议"""
    disposal_map = {
        "边坡滑坡": "建议立即采取边坡防护措施，必要时封闭交通，组织专业队伍进行治理。",
        "排水沟积水": "建议及时疏通排水沟，清理淤积物，恢复排水功能。",
        "排水沟破损": "建议修复排水沟破损部位，确保排水系统正常运行。"
    }
    return disposal_map.get(disease_type, f"建议及时处置{disease_type}，确保高速公路安全运营。")


# 初始化检查函数
def check_rag_availability() -> Dict[str, Any]:
    """检查RAG功能可用性

    Returns:
        包含状态信息的字典
    """
    result = {
        "langchain_available": LANGCHAIN_AVAILABLE,
        "embedding_model_exists": False,
        "knowledge_dir_exists": False,
        "vectorstore_loaded": False,
        "rag_enabled": False
    }

    if not LANGCHAIN_AVAILABLE:
        result["error"] = "langchain未安装，请运行: pip install langchain langchain-community"
        return result

    embedding_path = os.environ.get("EMBEDDING_MODEL_PATH", "./bge-small-zh-v1.5")
    result["embedding_model_exists"] = os.path.exists(embedding_path)
    result["embedding_model_path"] = embedding_path

    knowledge_dir = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")
    result["knowledge_dir_exists"] = os.path.exists(knowledge_dir)
    result["knowledge_dir_path"] = knowledge_dir

    # 检查全局实例
    global _knowledge_base
    if _knowledge_base is not None:
        result["vectorstore_loaded"] = _knowledge_base.is_available()
        result["rag_enabled"] = _knowledge_base.use_rag

    result["rag_enabled"] = (
        LANGCHAIN_AVAILABLE and
        result["embedding_model_exists"] and
        result["knowledge_dir_exists"]
    )

    return result