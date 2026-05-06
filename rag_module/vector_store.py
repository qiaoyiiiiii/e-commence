import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rag_module.data_processor import DataProcessor
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = self._initialize_embedding_model()
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_embedding_model(self):
        """ Initializes the HuggingFace BGE Embedding model. """
        logging.info(f"Initializing embedding model: {Config.EMBEDDING_MODEL_NAME} on device: {Config.EMBEDDING_MODEL_DEVICE}")
        return HuggingFaceBgeEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": Config.EMBEDDING_MODEL_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="为这个句子生成表示以用于检索相关文章："
        )

    def _initialize_vectorstore(self):
        """ Initializes or loads the Chroma vector store. """
        if not os.path.exists(Config.CHROMA_DB_PATH):
            logging.info(f"Chroma DB not found at {Config.CHROMA_DB_PATH}, creating new one...")
            # Load data from MySQL via DataProcessor
            data_processor = DataProcessor()
            documents = data_processor.load_and_process_goods()

            if not documents:
                logging.warning("No documents to add to vector store. Ensure goods data is in MySQL.")
                # Create an empty vectorstore if no documents, or handle as error
                return Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=Config.CHROMA_DB_PATH,
                    collection_name=Config.COLLECTION_NAME
                )

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=Config.CHROMA_DB_PATH,
                collection_name=Config.COLLECTION_NAME
            )
            vectorstore.persist()
            logging.info("Chroma DB created and persisted with goods data.")
        else:
            logging.info(f"Loading existing Chroma DB from {Config.CHROMA_DB_PATH}...")
            vectorstore = Chroma(
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=Config.COLLECTION_NAME
            )
            logging.info("Chroma DB loaded.")
        return vectorstore

    def get_retriever(self, search_kwargs: dict = None):
        """ Returns a retriever instance from the vector store. """
        if search_kwargs is None:
            search_kwargs = {"k": Config.RETRIEVER_K}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def add_documents(self, documents: List[Document]):
        """ Adds new documents to the vector store. """
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logging.info(f"Added {len(documents)} documents to the vector store.")
        else:
            logging.error("Vector store not initialized. Cannot add documents.")

# Example usage
if __name__ == "__main__":
    # Ensure you have some goods data in your MySQL DB and .env is configured
    print("Initializing VectorStoreManager...")
    vector_manager = VectorStoreManager()
    print("VectorStoreManager initialized.")

    # Test retrieval
    query = "轻便耐用的帆布包"
    print(f"\nRetrieving documents for query: '{query}'")
    retriever = vector_manager.get_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:\nPage Content: {doc.page_content}\nMetadata: {doc.metadata.get('name', '')}\n")

    # Example of adding new documents (if new goods are added to MySQL)
    # new_goods_data = [{'goods_id': 'G002', 'name': '时尚潮流双肩包', 'category': '包包', ...}]
    # new_documents = DataProcessor().goods_to_langchain_documents(new_goods_data)
    # if new_documents:
    #     vector_manager.add_documents(new_documents)

    print("VectorStoreManager example complete.")
