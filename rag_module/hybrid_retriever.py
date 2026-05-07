"""
模块：hybrid_retriever.py
职责：
    提供混合检索管理器（HybridRetrieverManager），整合向量检索与
    可选的 Cross-Encoder 重排序（Reranking）能力，对外暴露统一的
    get_retriever() 接口供 RAG 链使用。

检索策略：
    - 重排序已启用（Config.RERANKING_ENABLED=True）：
        使用 ContextualCompressionRetriever 将 Reranker 作为压缩器包裹
        在向量检索器之上，先粗检再精排，提升最终召回文档的相关性。
    - 重排序已禁用（Config.RERANKING_ENABLED=False）：
        直接返回 ChromaDB 向量检索器，性能更高但排序精度略低。

依赖：
    - langchain.retrievers.ContextualCompressionRetriever：LangChain 压缩检索器
    - rag_module.vector_store.VectorStoreManager：向量库与基础检索器
    - rag_module.reranker.Reranker：Cross-Encoder 重排序器
    - config.Config：全局配置（RERANKING_ENABLED、RETRIEVER_K 等）

使用方式：
    manager   = HybridRetrieverManager()
    retriever = manager.get_retriever()
    docs      = retriever.invoke("适合学生的便宜背包")

扩展说明：
    当前"混合"主要体现在"向量检索 + 重排序"的组合；
    未来可在此模块中叠加关键词检索（BM25 等）并实现 RRF 融合，
    真正实现稀疏+稠密的混合检索。
"""

from langchain.retrievers import ContextualCompressionRetriever
from rag_module.vector_store import VectorStoreManager
from rag_module.reranker import Reranker
from config import Config
import logging

# 初始化日志，使用 Config 中定义的日志级别和格式
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRetrieverManager:
    """
    混合检索管理器。

    职责：
        在 __init__ 阶段完成所有重型组件（向量库、嵌入模型、重排序模型）的
        一次性初始化，后续通过 get_retriever() 按需返回检索器实例，
        避免每次检索时重复加载模型带来的性能损耗。

    属性：
        vector_store_manager (VectorStoreManager)：向量库管理器，内含嵌入模型。
        base_retriever       (VectorStoreRetriever)：基础向量检索器，
            检索数量由 Config.RETRIEVER_K 控制。
        reranker_instance    (CrossEncoderReranker | None)：重排序器实例。
            若 Config.RERANKING_ENABLED=False，则为 None。
    """

    def __init__(self):
        """
        初始化混合检索管理器。

        执行顺序：
            1. 初始化 VectorStoreManager（加载嵌入模型 + ChromaDB）。
            2. 从向量库获取基础检索器（top-k 由 Config.RETRIEVER_K 决定）。
            3. 根据 Config.RERANKING_ENABLED 决定是否初始化 Reranker。
               Reranker 加载 Cross-Encoder 模型，耗时较长，
               仅在配置开启时才执行，避免不必要的资源占用。
        """
        # 初始化向量库管理器（内部完成嵌入模型加载和 ChromaDB 初始化）
        self.vector_store_manager = VectorStoreManager()

        # 获取基础向量检索器，检索候选数量由配置决定
        # 重排序场景下，RETRIEVER_K 通常设置得较大（如 20），以保证重排前有足够候选集
        self.base_retriever = self.vector_store_manager.get_retriever(search_kwargs={"k": Config.RETRIEVER_K})

        # 根据配置决定是否加载重排序模型
        self.reranker_instance = None
        if Config.RERANKING_ENABLED:
            # 加载 Cross-Encoder 重排序模型（首次加载会下载模型权重，耗时较长）
            self.reranker_instance = Reranker().get_reranker()
            logging.info("Reranking is enabled for hybrid retriever.")
        else:
            logging.info("Reranking is disabled for hybrid retriever.")

    def get_retriever(self):
        """
        返回配置好的检索器实例。

        根据初始化时 reranker_instance 是否为 None，动态选择返回策略：

        策略一（重排序已启用）：
            返回 ContextualCompressionRetriever，内部流程为：
                用户查询 → 向量检索（粗排，取 RETRIEVER_K 条）
                         → Cross-Encoder 重排序（精排，取 RERANKER_TOP_N 条）
                         → 返回相关性最高的 top-n 文档

        策略二（重排序已禁用）：
            直接返回 base_retriever（纯向量相似度检索），
            省去重排序开销，适合对延迟敏感的场景。

        返回：
            ContextualCompressionRetriever 或 VectorStoreRetriever：
                均实现了 LangChain Retriever 接口，可直接调用 .invoke(query)。
        """
        if self.reranker_instance:
            # 将 Reranker 作为"压缩器"包裹在向量检索器之上，
            # ContextualCompressionRetriever 先调用 base_retriever 粗检，
            # 再交由 base_compressor（即 Reranker）对候选文档重打分并截取 top-n
            retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker_instance,  # Cross-Encoder 重排序器
                base_retriever=self.base_retriever        # 向量粗检索器
            )
            logging.info("Returning ContextualCompressionRetriever with reranking.")
            return retriever
        else:
            # 未启用重排序，直接返回向量检索器
            logging.info("Returning base vector store retriever (reranking disabled).")
            return self.base_retriever

    # 未来扩展点：可在此添加关键词检索（BM25）并与向量检索结果做 RRF 融合，
    # 实现真正意义上的稀疏+稠密混合检索（Hybrid Retrieval）。


# ── 模块独立运行时的简单演示 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 运行前提：MySQL 已启动，数据库/表已创建，.env 配置正确
    print("Initializing HybridRetrieverManager...")
    hybrid_retriever_manager = HybridRetrieverManager()
    retriever = hybrid_retriever_manager.get_retriever()
    print("HybridRetrieverManager initialized.")

    query = "适合学生党用的便宜的包包"
    print(f"\nRetrieving documents for query: '{query}'")
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        # 重排序启用时，metadata 中会附带 relevance_score 字段
        score_info = f" (relevance_score: {doc.metadata['relevance_score']:.4f})" if Config.RERANKING_ENABLED else ""
        print(f"- {doc.page_content[:100]}... {score_info}\nMetadata Name: {doc.metadata.get('name', '')}\n")
