"""
模块：reranker.py
职责：
    封装基于 Cross-Encoder 的文档重排序（Reranking）功能。

    在 RAG 检索链中，向量检索（Bi-Encoder）能够高效地从大规模语料中
    快速召回候选文档，但其相关性排序精度有限。Cross-Encoder 重排序器
    对候选文档与查询做精细的交叉注意力打分，显著提升最终返回文档的
    相关性，以"精排"补充"粗排"的不足。

工作流程：
    用户查询 → 向量检索（粗排，取 top-K 候选）
             → Cross-Encoder 重打分（精排）
             → 截取 top-N 最相关文档返回给 LLM

依赖：
    - langchain_community.cross_encoders.HuggingFaceCrossEncoder：
        HuggingFace Cross-Encoder 模型封装
    - langchain.retrievers.document_compressors.CrossEncoderReranker：
        LangChain 重排序压缩器，与 ContextualCompressionRetriever 配合使用
    - config.Config：全局配置（模型名称、设备、top-n 数量）

使用方式：
    reranker = Reranker().get_reranker()
    # 通常不直接调用，而是传入 ContextualCompressionRetriever 使用：
    from langchain.retrievers import ContextualCompressionRetriever
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_vector_retriever
    )
"""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from config import Config
import logging

# 初始化日志，使用 Config 中定义的日志级别和格式
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class Reranker:
    """
    Cross-Encoder 重排序器封装类。

    职责：
        在初始化时加载 HuggingFace Cross-Encoder 模型，并构造
        LangChain CrossEncoderReranker 实例，供上层检索链调用。

    Cross-Encoder 原理：
        与 Bi-Encoder（嵌入模型）不同，Cross-Encoder 将"查询+文档"
        拼接后一起输入 Transformer，直接输出相关性分数，
        精度更高但推理速度较慢，因此通常只对粗排候选集（几十条）做精排。

    属性：
        cross_encoder_model (HuggingFaceCrossEncoder)：
            底层 Cross-Encoder 模型实例。
        reranker (CrossEncoderReranker)：
            LangChain 重排序压缩器实例，内部调用 cross_encoder_model
            对文档列表打分并截取 top-n 结果。
    """

    def __init__(self):
        """
        初始化 Reranker，加载 Cross-Encoder 模型并构建重排序器。

        配置项（来自 Config）：
            - RERANKER_MODEL_NAME   ：Cross-Encoder 模型名称或本地路径。
            - RERANKER_MODEL_DEVICE ：推理设备，如 "cpu" 或 "cuda"。
            - RERANKER_TOP_N        ：重排序后保留的文档数量（top-n）。

        说明：
            首次运行时若使用在线模型名称，会从 HuggingFace Hub 下载权重，
            耗时取决于网络速度和模型大小；后续运行直接读本地缓存。
        """
        # 加载 HuggingFace Cross-Encoder 模型
        # model_kwargs 传递给底层 sentence-transformers，用于指定推理设备
        self.cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=Config.RERANKER_MODEL_NAME,
            model_kwargs={"device": Config.RERANKER_MODEL_DEVICE}  # 支持 "cpu" / "cuda" / "mps"
        )

        # 构建 LangChain CrossEncoderReranker 压缩器
        # top_n：从粗排候选中最终保留的文档数量，需小于等于向量检索的 top-k
        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder_model,
            top_n=Config.RERANKER_TOP_N  # 精排后截取的最终文档数
        )
        logging.info(f"Reranker initialized with model: {Config.RERANKER_MODEL_NAME}, top_n: {Config.RERANKER_TOP_N}")

    def get_reranker(self):
        """
        返回已初始化的 CrossEncoderReranker 实例。

        通常将返回值传入 ContextualCompressionRetriever 的
        base_compressor 参数，由 LangChain 框架在检索链中自动调用。

        返回：
            CrossEncoderReranker：LangChain 兼容的文档压缩器实例，
                实现了 compress_documents(documents, query) 接口。
        """
        return self.reranker


# ── 模块独立运行时的简单演示 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 首次运行需要联网下载模型权重；后续运行直接读本地缓存
    print("Initializing Reranker...")
    reranker_instance = Reranker().get_reranker()
    print("Reranker initialized.")

    # 构造测试文档，模拟向量检索粗排后的候选集
    from langchain_core.documents import Document
    docs = [
        Document(page_content="这是一个关于红色连衣裙的描述。", metadata={"source": "doc1"}),
        Document(page_content="我喜欢蓝色的T恤和牛仔裤。", metadata={"source": "doc2"}),
        Document(page_content="这款连衣裙非常适合夏天穿。", metadata={"source": "doc3"}),
    ]
    query = "我想找一件漂亮的连衣裙"
    print(f"\nQuery: {query}")
    print("Original documents:")
    for doc in docs:
        print(f"- {doc.page_content}")

    # 调用重排序：对候选文档重新打分并按相关性排序，截取 top-n 结果
    reranked_docs = reranker_instance.compress_documents(documents=docs, query=query)
    print(f"\nReranked documents (top {Config.RERANKER_TOP_N}):")
    for doc in reranked_docs:
        # compress_documents 会在 metadata 中注入 relevance_score 字段
        print(f"- {doc.page_content} (relevance_score: {doc.metadata['relevance_score']:.4f})")
