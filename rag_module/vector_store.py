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
    4. 支持全量重置向量库（sync_from_mysql）。

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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
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

            data_processor = DataProcessor()
            documents = data_processor.load_and_process_goods()

            if not documents:
                # MySQL 暂无商品数据：先建空库保证系统可正常启动，
                # 后续商品入库后调用 sync_from_mysql() 重建向量索引
                logging.warning(
                    "No documents found in MySQL. Creating empty vector store. "
                    "Call sync_from_mysql() after populating MySQL to index goods."
                )
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
            logging.info(f"Chroma DB created with {len(documents)} documents.")
        else:
            # ── 非首次运行：直接加载已有向量库 ───────────────────────────
            logging.info(f"Loading existing Chroma DB from {Config.CHROMA_DB_PATH}...")
            vectorstore = Chroma(
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=Config.COLLECTION_NAME
            )
            logging.info("Chroma DB loaded.")

            # ── 自动补全：向量库为空但 MySQL 已有数据时自动填充 ───────────
            # 场景：首次启动时 MySQL 为空建了空库，之后 MySQL 导入了商品数据，
            # 下次重启应自动将这批数据索引进向量库，无需手动干预
            doc_count = vectorstore._collection.count()
            if doc_count == 0:
                logging.info(
                    "Vector store is empty. Attempting to auto-populate from MySQL..."
                )
                try:
                    documents = DataProcessor().load_and_process_goods()
                    if documents:
                        vectorstore.add_documents(documents)
                        logging.info(
                            f"Auto-populated vector store with {len(documents)} documents."
                        )
                    else:
                        logging.info("MySQL also empty; vector store remains empty.")
                except Exception as e:
                    logging.warning(f"Auto-populate failed (system will continue): {e}")

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

    def sync_from_mysql(self) -> int:
        """
        全量同步：清空向量库并以 MySQL 当前商品数据重建索引。

        适用场景：
            - 商品信息（名称、描述、标签）批量更新后需要刷新向量索引。
            - MySQL 导入了大批新商品，且不希望出现向量库中存在已下架商品的情况。

        实现方式：
            删除整个 ChromaDB Collection 后重建，再批量写入最新的商品文档。
            因此，调用期间检索器会短暂返回空结果；生产环境建议在低峰期执行。

        返回：
            int：成功写入向量库的文档数量；MySQL 无数据时返回 0。

        副作用：
            - 向量库中原有的所有文档将被删除并以最新数据替换。
            - 操作完成后 self.vectorstore 指向重建后的新集合实例。
        """
        logging.info("Starting full sync: dropping existing collection and rebuilding from MySQL...")
        try:
            documents = DataProcessor().load_and_process_goods()
        except Exception as e:
            logging.error(f"sync_from_mysql: failed to load documents from MySQL: {e}")
            return 0

        if not documents:
            logging.warning("sync_from_mysql: no documents found in MySQL, sync aborted.")
            return 0

        # 删除旧集合（包含所有向量数据），然后重建空集合
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            persist_directory=Config.CHROMA_DB_PATH,
            embedding_function=self.embedding_model,
            collection_name=Config.COLLECTION_NAME,
        )

        # 批量写入最新文档
        self.vectorstore.add_documents(documents)
        logging.info(f"sync_from_mysql: rebuilt vector store with {len(documents)} documents.")
        return len(documents)


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

    # 全量重置示例（新增或更新商品后使用）：
    # count = vector_manager.sync_from_mysql()
    # print(f"Synced {count} documents from MySQL.")

    print("VectorStoreManager example complete.")
