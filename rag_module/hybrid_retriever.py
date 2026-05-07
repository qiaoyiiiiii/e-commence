"""
模块：hybrid_retriever.py
职责：
    提供混合检索管理器（HybridRetrieverManager），将稀疏检索（BM25）与
    稠密检索（向量相似度）通过 RRF（Reciprocal Rank Fusion）融合，
    并可选地在融合结果之上叠加 Cross-Encoder 重排序（Reranking）。

完整检索流水线：
    ┌─────────────────────────────────────────────────────┐
    │  用户查询                                            │
    │     ↓                    ↓                          │
    │  [BM25 稀疏检索]    [向量稠密检索]                   │
    │  (关键词匹配，       (语义相似度，                    │
    │  取 RETRIEVER_K 条)  取 RETRIEVER_K 条)              │
    │          ↓                    ↓                     │
    │     [EnsembleRetriever  RRF 融合排序]                │
    │          ↓                                          │
    │  [可选：Cross-Encoder Reranker 精排]                 │
    │          ↓                                          │
    │  最终 top-n 文档                                    │
    └─────────────────────────────────────────────────────┘

配置开关：
    - Config.BM25_ENABLED       : 是否启用 BM25 稀疏检索（默认 True）
    - Config.RERANKING_ENABLED  : 是否启用 Cross-Encoder 重排序（默认 True）
    - Config.ENSEMBLE_VECTOR_WEIGHT : 向量检索在 RRF 融合中的权重（默认 0.5）
    - Config.RETRIEVER_K        : 各路检索器召回候选文档数量（默认 20）

依赖：
    - langchain.retrievers.ContextualCompressionRetriever : 压缩检索器（含 Reranker）
    - langchain.retrievers.EnsembleRetriever              : RRF 融合检索器
    - langchain_community.retrievers.BM25Retriever        : BM25 稀疏检索器
    - rag_module.vector_store.VectorStoreManager          : 向量库与基础检索器
    - rag_module.reranker.Reranker                        : Cross-Encoder 重排序器
    - rag_module.data_processor.DataProcessor             : 从 MySQL 加载文档供 BM25 索引
    - config.Config                                       : 全局配置

使用方式：
    manager   = HybridRetrieverManager()
    retriever = manager.get_retriever()
    docs      = retriever.invoke("适合学生的便宜背包")
"""

import logging
from typing import List, Optional

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_module.vector_store import VectorStoreManager
from rag_module.reranker import Reranker
from rag_module.data_processor import DataProcessor
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


def _chinese_char_tokenizer(text: str) -> List[str]:
    """
    中文字符级分词器，供 BM25Retriever 使用。

    BM25Retriever 默认以空格分词，对中文几乎失效（中文词之间无空格）。
    将文本拆分为单个字符后，BM25 可在字符粒度上做关键词匹配，
    对中文短查询（如商品名、标签）有较好的召回效果。

    参数：
        text (str): 待分词的文本字符串。

    返回：
        List[str]: 字符列表，例如 "通勤包" → ["通", "勤", "包"]。

    注意：
        若项目后续引入 jieba 等中文分词库，可将此函数替换为
        ``jieba.lcut(text)`` 以获得更精准的词级分词效果。
    """
    return list(text)


class HybridRetrieverManager:
    """
    混合检索管理器：稀疏（BM25）+ 稠密（向量）双路召回，RRF 融合，可选 Reranker 精排。

    初始化时一次性完成所有重型组件（向量库、BM25 索引、重排序模型）的加载，
    后续通过 get_retriever() 返回配置好的检索器实例，避免重复加载模型。

    属性：
        vector_store_manager (VectorStoreManager)  : 向量库管理器（含嵌入模型）。
        vector_retriever                           : 基础向量检索器（ChromaDB）。
        bm25_retriever (BM25Retriever | None)      : BM25 稀疏检索器；
            BM25_ENABLED=False 或文档为空时为 None。
        ensemble_retriever (EnsembleRetriever|None): RRF 融合检索器；
            BM25 不可用时为 None，退化为纯向量检索。
        reranker_instance                          : Cross-Encoder 重排序器；
            RERANKING_ENABLED=False 时为 None。
    """

    def __init__(self):
        """
        按以下顺序初始化各检索组件：
          1. 向量检索器（ChromaDB + 嵌入模型）
          2. BM25 稀疏检索器（从 MySQL 加载文档，建立内存索引）
          3. EnsembleRetriever（RRF 融合，仅当 BM25 可用时创建）
          4. Reranker（Cross-Encoder 重排序，按配置开关决定是否加载）
        """
        # ── 步骤 1：初始化向量检索器 ────────────────────────────────────────
        self.vector_store_manager = VectorStoreManager()
        # 重排序场景下 RETRIEVER_K 通常设大（如 20），保证重排前有足够候选集
        self.vector_retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": Config.RETRIEVER_K}
        )
        logging.info(f"Vector retriever initialized (k={Config.RETRIEVER_K}).")

        # ── 步骤 2：初始化 BM25 稀疏检索器 ────────────────────────────────
        self.bm25_retriever: Optional[BM25Retriever] = None
        if Config.BM25_ENABLED:
            documents = self._load_documents_for_bm25()
            if documents:
                # preprocess_func 使用字符级分词，适配中文文本
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents,
                    preprocess_func=_chinese_char_tokenizer,
                    k=Config.RETRIEVER_K,
                )
                logging.info(
                    f"BM25 retriever initialized with {len(documents)} documents "
                    f"(k={Config.RETRIEVER_K})."
                )
            else:
                logging.warning(
                    "BM25 enabled but no documents loaded from MySQL; "
                    "falling back to vector-only retrieval."
                )
        else:
            logging.info("BM25 retrieval is disabled by config.")

        # ── 步骤 3：构建 EnsembleRetriever（RRF 融合） ─────────────────────
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        if self.bm25_retriever is not None:
            vector_weight = Config.ENSEMBLE_VECTOR_WEIGHT
            bm25_weight = round(1.0 - vector_weight, 4)
            # EnsembleRetriever 内置 RRF（c=60），weights 决定各路结果在融合分中的权重
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[bm25_weight, vector_weight],
            )
            logging.info(
                f"EnsembleRetriever (RRF) initialized: "
                f"BM25 weight={bm25_weight}, Vector weight={vector_weight}."
            )

        # ── 步骤 4：初始化 Reranker（可选精排） ────────────────────────────
        self.reranker_instance = None
        if Config.RERANKING_ENABLED:
            self.reranker_instance = Reranker().get_reranker()
            logging.info("Cross-Encoder reranker initialized.")
        else:
            logging.info("Reranking is disabled by config.")

    def _load_documents_for_bm25(self) -> List[Document]:
        """
        从 MySQL 加载商品文档，用于在内存中构建 BM25 索引。

        调用 DataProcessor.load_and_process_goods()，其返回的 Document 列表
        与 ChromaDB 向量库中存储的文档内容一致（page_content 为结构化中文文本）。

        返回：
            List[Document]: 商品文档列表；加载失败或无数据时返回空列表。

        注意：
            BM25 索引驻留在内存中，不做持久化；每次启动均需重新加载。
            对于商品量较大（万级以上）的场景，可考虑将文档缓存到本地文件
            以减少启动时的 MySQL 查询开销。
        """
        try:
            processor = DataProcessor()
            documents = processor.load_and_process_goods()
            return documents
        except Exception as e:
            logging.warning(f"Failed to load documents for BM25 retriever: {e}")
            return []

    def get_retriever(self):
        """
        返回配置好的完整检索器实例。

        按以下优先级选择基础检索器：
          - BM25 可用 → EnsembleRetriever（RRF 融合）作为基础
          - BM25 不可用 → 退化为纯向量检索器

        若 Reranker 已启用，则在基础检索器之上再叠加
        ContextualCompressionRetriever 进行 Cross-Encoder 精排。

        最终返回的检索器统一实现 LangChain Retriever 接口，
        可直接调用 .invoke(query) 或传入 create_retrieval_chain。

        返回：
            Retriever：具体类型取决于配置，可能为：
                - EnsembleRetriever（仅 BM25+向量，无精排）
                - VectorStoreRetriever（仅向量，无精排）
                - ContextualCompressionRetriever（上述任一 + Reranker 精排）
        """
        # 优先使用 RRF 融合检索器；BM25 不可用时退化为纯向量
        base_retriever = self.ensemble_retriever if self.ensemble_retriever else self.vector_retriever

        if self.reranker_instance:
            # 用 ContextualCompressionRetriever 将 Reranker 包裹在基础检索器之上：
            #   base_retriever 先粗检（BM25+向量 RRF 或纯向量）
            #   base_compressor（Reranker）再对候选文档重打分并截取 top-n
            retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker_instance,
                base_retriever=base_retriever,
            )
            mode = "BM25+Vector(RRF)" if self.ensemble_retriever else "Vector-only"
            logging.info(f"Returning [{mode}] + Reranker retriever.")
            return retriever
        else:
            mode = "BM25+Vector(RRF)" if self.ensemble_retriever else "Vector-only"
            logging.info(f"Returning [{mode}] retriever (no reranking).")
            return base_retriever


# ── 模块独立运行时的简单演示 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 运行前提：MySQL 已启动，数据库/表已创建，rank_bm25 已安装，.env 配置正确
    print("Initializing HybridRetrieverManager...")
    manager = HybridRetrieverManager()
    retriever = manager.get_retriever()
    print("HybridRetrieverManager initialized.\n")

    query = "适合学生党用的便宜包包"
    print(f"Query: '{query}'")
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for doc in retrieved_docs:
        # 重排序启用时，metadata 中会附带 relevance_score 字段
        score = (
            f" (relevance_score: {doc.metadata['relevance_score']:.4f})"
            if Config.RERANKING_ENABLED and 'relevance_score' in doc.metadata
            else ""
        )
        print(f"- {doc.metadata.get('name', '未知')}{score}")
        print(f"  {doc.page_content[:80].strip()}...\n")
