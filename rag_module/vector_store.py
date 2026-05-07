"""
模块：vector_store.py
职责：
    管理商品向量数据库（ChromaDB）的初始化、加载与检索。

    系统首次运行时，若本地不存在持久化的 ChromaDB 目录，
    则自动从 MySQL 读取商品数据、生成向量嵌入并写入新建的向量库；
    后续运行直接加载已持久化的向量库，避免重复建库。

主要功能：
    1. 初始化或加载 HuggingFace BGE 嵌入模型。
    2. 初始化或加载 ChromaDB 持久化向量库。
    3. 对外暴露 get_retriever() 接口，供检索链使用。
    4. 支持动态追加新文档（add_documents）。

依赖：
    - langchain_community.embeddings.HuggingFaceBgeEmbeddings：BGE 嵌入模型封装
    - langchain_community.vectorstores.Chroma：ChromaDB 向量库封装
    - rag_module.data_processor.DataProcessor：商品数据加载与转换
    - config.Config：全局配置（模型路径、设备、ChromaDB 路径、检索 top-k 等）

使用方式：
    manager   = VectorStoreManager()
    retriever = manager.get_retriever()           # 获取默认 top-k 检索器
    retriever = manager.get_retriever({"k": 10})  # 获取自定义 top-k 检索器
"""

import os
from typing import List
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rag_module.data_processor import DataProcessor
from config import Config
import logging

# 初始化日志，使用 Config 中定义的日志级别和格式
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorStoreManager:
    """
    向量数据库管理器。

    封装了 BGE 嵌入模型与 ChromaDB 向量库的完整生命周期管理：
        - 初始化时自动判断向量库是否已存在；
        - 不存在则从 MySQL 拉取商品数据并新建向量库；
        - 已存在则直接加载，跳过耗时的重建流程。

    属性：
        embedding_model (HuggingFaceBgeEmbeddings)：用于将文本转换为向量的嵌入模型。
        vectorstore     (Chroma)：ChromaDB 向量数据库实例。
    """

    def __init__(self):
        """
        初始化 VectorStoreManager。

        执行顺序：
            1. 加载嵌入模型（_initialize_embedding_model）。
            2. 初始化或加载向量库（_initialize_vectorstore）。
               向量库初始化依赖嵌入模型，因此顺序不可颠倒。
        """
        # 第一步：加载嵌入模型，后续建库和检索均依赖此模型
        self.embedding_model = self._initialize_embedding_model()
        # 第二步：基于嵌入模型初始化且加载向量库
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_embedding_model(self):
        """
        初始化 HuggingFace BGE 系列嵌入模型。

        BGE（BAAI General Embedding）是由北京智源研究院开源的中英双语嵌入模型，
        适合中文电商场景的语义检索任务。

        配置项（来自 Config）：
            - EMBEDDING_MODEL_NAME  ：模型名称或本地路径。
            - EMBEDDING_MODEL_DEVICE：推理设备，如 "cpu" 或 "cuda"。

        返回：
            HuggingFaceBgeEmbeddings：已初始化的嵌入模型实例。

        说明：
            - normalize_embeddings=True：对输出向量做 L2 归一化，
              保证余弦相似度计算的数值稳定性。
            - query_instruction：BGE 模型推荐在查询文本前添加指令前缀，
              以提升检索精度（文档侧不需要该前缀）。
        """
        logging.info(f"Initializing embedding model: {Config.EMBEDDING_MODEL_NAME} on device: {Config.EMBEDDING_MODEL_DEVICE}")
        return HuggingFaceBgeEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": Config.EMBEDDING_MODEL_DEVICE},
            encode_kwargs={"normalize_embeddings": True},   # L2 归一化，提升余弦相似度稳定性
            query_instruction="为这个句子生成表示以用于检索相关文章："  # BGE 官方推荐查询指令前缀
        )

    def _initialize_vectorstore(self):
        """
        初始化或加载 ChromaDB 持久化向量库。

        逻辑分支：
            - 若 Config.CHROMA_DB_PATH 目录不存在（首次运行）：
                1. 通过 DataProcessor 从 MySQL 加载商品数据并转换为 Document。
                2. 若无商品数据，创建空向量库并返回（后续可通过 add_documents 填充）。
                3. 否则调用 Chroma.from_documents 批量嵌入并持久化。
            - 若目录已存在（非首次运行）：
                直接加载已持久化的向量库，跳过重新建库流程。

        返回：
            Chroma：已初始化的 ChromaDB 向量库实例。

        副作用：
            首次运行会在 Config.CHROMA_DB_PATH 路径下创建持久化文件。
        """
        if not os.path.exists(Config.CHROMA_DB_PATH):
            # ── 首次运行：从零创建向量库 ──────────────────────────────────
            logging.info(f"Chroma DB not found at {Config.CHROMA_DB_PATH}, creating new one...")

            # 创建数据处理器-》获取db数据库实例-》加载货物-》documents[content, metadata]
            data_processor = DataProcessor()
            documents = data_processor.load_and_process_goods()

            if not documents:
                # 数据库中暂无商品数据，创建空向量库以保证系统可正常启动；
                # 后续可通过 add_documents 方法动态补充文档
                logging.warning("No documents to add to vector store. Ensure goods data is in MySQL.")
                return Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=Config.CHROMA_DB_PATH,
                    collection_name=Config.COLLECTION_NAME
                )

            # 批量嵌入文档并写入 ChromaDB，同时持久化到磁盘
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=Config.CHROMA_DB_PATH,
                collection_name=Config.COLLECTION_NAME
            )
            logging.info("Chroma DB created and persisted with goods data.")
        else:
            # ── 非首次运行：直接加载已有向量库 ───────────────────────────
            logging.info(f"Loading existing Chroma DB from {Config.CHROMA_DB_PATH}...")
            vectorstore = Chroma(
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=Config.COLLECTION_NAME
            )
            logging.info("Chroma DB loaded.")

        return vectorstore

    def get_retriever(self, search_kwargs: dict = None):
        """
        从向量库创建并返回检索器实例。

        参数：
            search_kwargs (dict, 可选)：传递给 as_retriever 的检索参数。
                常用键：
                    - "k" (int)：返回的最相似文档数量，默认取 Config.RETRIEVER_K。
                若不传则使用配置文件中的默认值。

        返回：
            VectorStoreRetriever：LangChain 检索器实例，
                可直接用于 RetrievalQA 链或 invoke(query) 调用。
        """
        if search_kwargs is None:
            # 未指定检索参数时，使用配置文件中的默认 top-k 值
            search_kwargs = {"k": Config.RETRIEVER_K}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def add_documents(self, documents: List[Document]):
        """
        向已初始化的向量库中追加新文档。

        适用场景：新商品上架后无需重建整个向量库，
        只需将新商品转换为 Document 并调用此方法增量写入。

        参数：
            documents (List[Document])：待追加的 LangChain Document 列表，
                metadata 应已经过 filter_complex_metadata 处理。

        副作用：
            - 成功时以 INFO 级别记录追加数量。
            - 向量库未初始化时以 ERROR 级别记录错误，不抛出异常。
        """
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            logging.info(f"Added {len(documents)} documents to the vector store.")
        else:
            # 正常情况下 __init__ 保证 vectorstore 已初始化，此分支为防御性处理
            logging.error("Vector store not initialized. Cannot add documents.")


# ── 模块独立运行时的简单演示 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 运行前提：MySQL 已启动，数据库/表已创建，.env 配置正确
    print("Initializing VectorStoreManager...")
    vector_manager = VectorStoreManager()
    print("VectorStoreManager initialized.")

    # 测试语义检索
    query = "轻便耐用的帆布包"
    print(f"\nRetrieving documents for query: '{query}'")
    retriever = vector_manager.get_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:\nPage Content: {doc.page_content}\nMetadata: {doc.metadata.get('name', '')}\n")

    # 增量追加文档示例（新商品上架时使用）：
    # new_goods_data = [{'goods_id': 'G002', 'name': '时尚潮流双肩包', 'category': '包包', ...}]
    # new_documents = DataProcessor().goods_to_langchain_documents(new_goods_data)
    # if new_documents:
    #     vector_manager.add_documents(new_documents)

    print("VectorStoreManager example complete.")
