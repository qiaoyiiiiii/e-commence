"""
模块职责：
    集中管理整个电商购物 Agent 项目的所有配置项。
    配置项优先从环境变量读取，若环境变量未设置则使用硬编码默认值。
    支持通过项目根目录下的 .env 文件注入环境变量，方便本地开发与生产部署切换。

依赖：
    - python-dotenv：用于解析 .env 文件并将其中的键值对注入到 os.environ
    - os：读取环境变量

使用方式：
    from config import Config

    api_key = Config.DEEPSEEK_API_KEY
    db_host  = Config.MYSQL_HOST
"""

import os
from dotenv import load_dotenv

# 在模块导入时立即加载 .env 文件中的环境变量，
# 使得后续所有 os.getenv() 调用都能感知到 .env 中的配置。
load_dotenv()


class Config:
    """
    项目全局配置类。

    所有属性均为类属性（不需要实例化），直接通过 Config.XXX 访问。
    配置分为以下几个区域：
        1. 通用项目配置
        2. Hugging Face 模型缓存路径
        3. DeepSeek 大语言模型配置
        4. MySQL 数据库连接配置
        5. ChromaDB 向量数据库配置
        6. 重排序（Reranker）模型配置
        7. Agent 行为配置
        8. 记忆压缩策略配置
        9. 本地数据文件路径
        10. 日志配置
        11. LangChain 链路追踪配置
    """

    # -------------------------------------------------------------------------
    # 1. 通用项目配置
    # -------------------------------------------------------------------------

    # 项目名称，用于日志标题、LangChain 项目标识等展示场景
    PROJECT_NAME = "E-commerce Shopping Agent"

    # 日志级别，默认 INFO；可设置为 DEBUG / WARNING / ERROR 等
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # 调试模式开关；环境变量值为字符串，需显式转为布尔值
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    # -------------------------------------------------------------------------
    # 2. Hugging Face 模型缓存路径
    # -------------------------------------------------------------------------

    # 本地 Hugging Face 模型缓存目录，避免重复下载大模型文件
    # 若部署在 Linux 服务器上，可在 .env 中将此值改为对应路径
    HF_HOME = "D:\\hf_cache"

    if HF_HOME:
        # 将缓存目录同步写入多个环境变量，保证不同库（transformers、
        # sentence-transformers）都使用同一目录缓存模型权重
        os.environ["HF_HOME"] = HF_HOME
        os.environ["TRANSFORMERS_CACHE"] = HF_HOME          # transformers 库缓存目录
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_HOME  # sentence-transformers 缓存目录

    # -------------------------------------------------------------------------
    # 3. DeepSeek 大语言模型（LLM）配置
    # -------------------------------------------------------------------------

    # DeepSeek API 密钥，必须在 .env 或系统环境变量中配置，不可留空
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # 使用的 DeepSeek 模型名称；也可切换为 "deepseek-reasoner" 等推理模型
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

    # 采样温度：0.0 表示输出最确定（贪心），值越大输出越随机
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

    # -------------------------------------------------------------------------
    # 4. MySQL 数据库连接配置
    # -------------------------------------------------------------------------

    # 数据库主机地址
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")

    # 数据库登录用户名
    MYSQL_USER = os.getenv("MYSQL_USER", "root")

    # 数据库登录密码（生产环境务必通过环境变量注入，不要硬编码）
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123123")

    # 目标数据库名称；首次运行时会自动创建（见 database.py）
    MYSQL_DB = os.getenv("MYSQL_DB", "ecommerce_agent")

    # MySQL 服务端口，默认 3306
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))

    # -------------------------------------------------------------------------
    # 5. ChromaDB 向量数据库配置
    # -------------------------------------------------------------------------

    # ChromaDB 持久化存储路径；首次运行后向量数据将保存在此目录
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # ChromaDB 集合（Collection）名称，对应存储商品向量的命名空间
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ecommerce_goods")

    # 向量嵌入模型名称；使用 BAAI/bge-large-zh-v1.5 以获得更优中文语义效果
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")

    # 嵌入模型推理设备；有 GPU 时可改为 "cuda" 以提升速度
    EMBEDDING_MODEL_DEVICE = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")

    # 向量检索阶段返回的候选文档数量（重排序前的粗召回 Top-K）
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", 20))

    # -------------------------------------------------------------------------
    # 6. 重排序（Reranker）模型配置
    # -------------------------------------------------------------------------

    # 重排序模型名称；用于对粗召回结果进行精细排序，提升推荐精准度
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")

    # 重排序模型推理设备；有 GPU 时可改为 "cuda"
    RERANKER_MODEL_DEVICE = os.getenv("RERANKER_MODEL_DEVICE", "cpu")

    # 重排序后保留的文档数量（最终进入 LLM 上下文的 Top-N）
    RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 5))

    # -------------------------------------------------------------------------
    # 7. Agent 行为配置
    # -------------------------------------------------------------------------

    # 每次对话最多向用户推荐的商品数量
    RECOMMENDATION_COUNT = int(os.getenv("RECOMMENDATION_COUNT", 3))

    # 是否启用自我反思（Self-Reflection）：Agent 对自身回答进行二次审查
    SELF_REFLECTION_ENABLED = os.getenv("SELF_REFLECTION_ENABLED", "True").lower() == "true"

    # 是否启用重排序（Reranking）：对检索结果进行精排后再传入 LLM
    RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "True").lower() == "true"

    # 是否启用 BM25 稀疏检索（关键词匹配），与向量检索做 RRF 融合
    # 关闭后退化为纯向量检索；开启时会从 MySQL 额外加载一次文档建立内存索引
    BM25_ENABLED = os.getenv("BM25_ENABLED", "True").lower() == "true"

    # EnsembleRetriever RRF 融合时向量检索的权重（0~1）
    # BM25 权重 = 1 - ENSEMBLE_VECTOR_WEIGHT；两者各 0.5 时等权融合
    ENSEMBLE_VECTOR_WEIGHT = float(os.getenv("ENSEMBLE_VECTOR_WEIGHT", "0.5"))

    # -------------------------------------------------------------------------
    # 8. 记忆压缩策略配置
    # -------------------------------------------------------------------------

    # 短期记忆消息条数阈值：超过此数量时触发摘要压缩，防止上下文过长
    MEMORY_COMPRESSION_THRESHOLD = int(os.getenv("MEMORY_COMPRESSION_THRESHOLD", 10))

    # 压缩后保留的最近原始消息条数（其余转为摘要）
    MEMORY_COMPRESSION_KEEP_RECENT = int(os.getenv("MEMORY_COMPRESSION_KEEP_RECENT", 4))

    # -------------------------------------------------------------------------
    # 9. 本地数据文件路径（用于初始数据导入/种子数据）
    # -------------------------------------------------------------------------

    # 商品数据 JSON 文件路径，用于首次向数据库和向量库导入商品信息
    GOODS_DATA_PATH = os.getenv("GOODS_DATA_PATH", "./data/goods_data.json")

    # 标签库 JSON 文件路径，用于初始化标签分类体系
    TAG_LIBRARY_PATH = os.getenv("TAG_LIBRARY_PATH", "./data/tag_library.json")

    # -------------------------------------------------------------------------
    # 10. 日志配置
    # -------------------------------------------------------------------------

    # Agent 运行轨迹日志文件路径，用于离线排查问题
    LOG_FILE = os.getenv("LOG_FILE", "./logs/agent_trace.log")

    # -------------------------------------------------------------------------
    # 11. LangChain 链路追踪（LangSmith）配置
    # -------------------------------------------------------------------------

    # 是否开启 LangChain V2 追踪，开启后调用链将上报至 LangSmith 平台
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    # LangSmith 平台 API Key，追踪功能开启时必须配置
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # LangSmith 项目名称，用于在平台上区分不同项目的追踪数据
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", PROJECT_NAME)

    # LangSmith API 端点，一般无需修改
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


# 直接运行此文件时，打印关键配置项用于快速核查环境是否配置正确
if __name__ == "__main__":
    print("--- Configuration Check ---")
    print(f"DeepSeek API Key: {Config.DEEPSEEK_API_KEY}")
    print(f"MySQL Host: {Config.MYSQL_HOST}")
    print(f"Chroma DB Path: {Config.CHROMA_DB_PATH}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print(f"Reranker Model: {Config.RERANKER_MODEL_NAME}")
    print(f"Debug Mode: {Config.DEBUG_MODE}")
