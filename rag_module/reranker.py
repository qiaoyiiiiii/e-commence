from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class Reranker:
    def __init__(self):
        self.cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=Config.RERANKER_MODEL_NAME,
            model_kwargs={"device": Config.RERANKER_MODEL_DEVICE}
        )
        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder_model,
            top_n=Config.RERANKER_TOP_N
        )
        logging.info(f"Reranker initialized with model: {Config.RERANKER_MODEL_NAME}, top_n: {Config.RERANKER_TOP_N}")

    def get_reranker(self):
        """ Returns the initialized CrossEncoderReranker instance. """
        return self.reranker

# Example usage
if __name__ == "__main__":
    # This requires an active internet connection to download the model weights the first time.
    print("Initializing Reranker...")
    reranker_instance = Reranker().get_reranker()
    print("Reranker initialized.")

    # Dummy documents for testing
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

    # Rerank
    reranked_docs = reranker_instance.compress_documents(documents=docs, query=query)
    print(f"\nReranked documents (top {Config.RERANKER_TOP_N}):")
    for doc in reranked_docs:
        print(f"- {doc.page_content} (relevance_score: {doc.metadata['relevance_score']:.4f})")
