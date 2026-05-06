from langchain.retrievers import ContextualCompressionRetriever
from rag_module.vector_store import VectorStoreManager
from rag_module.reranker import Reranker
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRetrieverManager:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.base_retriever = self.vector_store_manager.get_retriever(search_kwargs={"k": Config.RETRIEVER_K})

        self.reranker_instance = None
        if Config.RERANKING_ENABLED:
            self.reranker_instance = Reranker().get_reranker()
            logging.info("Reranking is enabled for hybrid retriever.")
        else:
            logging.info("Reranking is disabled for hybrid retriever.")

    def get_retriever(self):
        """ Returns a configured retriever (with or without reranking). """
        if self.reranker_instance:
            # Combine the base retriever with the reranker
            retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker_instance,
                base_retriever=self.base_retriever
            )
            logging.info("Returning ContextualCompressionRetriever with reranking.")
            return retriever
        else:
            logging.info("Returning base vector store retriever (reranking disabled).")
            return self.base_retriever

    # Future enhancement: add keyword-based retrieval and merge logic here

# Example usage
if __name__ == "__main__":
    # Ensure you have some goods data in your MySQL DB and .env is configured
    print("Initializing HybridRetrieverManager...")
    hybrid_retriever_manager = HybridRetrieverManager()
    retriever = hybrid_retriever_manager.get_retriever()
    print("HybridRetrieverManager initialized.")

    query = "适合学生党用的便宜的包包"
    print(f"\nRetrieving documents for query: '{query}'")
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        score_info = f" (relevance_score: {doc.metadata['relevance_score']:.4f})" if Config.RERANKING_ENABLED else ""
        print(f"- {doc.page_content[:100]}... {score_info}\nMetadata Name: {doc.metadata.get('name', '')}\n")
