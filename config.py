import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # --- General Configuration ---
    PROJECT_NAME = "E-commerce Shopping Agent"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    # --- Hugging Face Cache Configuration ---
    # Set HF_HOME to change the default cache directory for Hugging Face models.
    # You can set this in your .env file, e.g., HF_HOME="D:\\hf_cache"
    HF_HOME = os.getenv("HF_HOME")
    if HF_HOME:
        os.environ["HF_HOME"] = HF_HOME
        # Also set TRANSFORMERS_CACHE and SENTENCE_TRANSFORMERS_HOME for compatibility
        os.environ["TRANSFORMERS_CACHE"] = HF_HOME
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_HOME

    # --- LLM Configuration ---
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat") # Or deepseek-reasoner
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    # Add other LLM specific configurations here if needed

    # --- MySQL Database Configuration ---
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123123")
    MYSQL_DB = os.getenv("MYSQL_DB", "ecommerce_agent")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))

    # --- ChromaDB Vector Store Configuration ---
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ecommerce_goods")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_MODEL_DEVICE = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu") # or "cpu"
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", 20)) # Number of documents to retrieve before reranking

    # --- Reranker Configuration ---
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
    RERANKER_MODEL_DEVICE = os.getenv("RERANKER_MODEL_DEVICE", "cpu") # or "cpu"
    RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 5)) # Number of documents after reranking

    # --- Agent Specific Configuration ---
    RECOMMENDATION_COUNT = int(os.getenv("RECOMMENDATION_COUNT", 3))
    SELF_REFLECTION_ENABLED = os.getenv("SELF_REFLECTION_ENABLED", "True").lower() == "true"
    RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "True").lower() == "true"

    # --- Data Paths (for initial loading if needed, though most will be DB driven) ---
    # This might be used for initial seeding of the database from local files before switching to full DB usage
    GOODS_DATA_PATH = os.getenv("GOODS_DATA_PATH", "./data/goods_data.json")
    TAG_LIBRARY_PATH = os.getenv("TAG_LIBRARY_PATH", "./data/tag_library.json")

    # --- Logging Configuration ---
    LOG_FILE = os.getenv("LOG_FILE", "./logs/agent_trace.log")

    # Langchain tracing for debugging (optional)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", PROJECT_NAME)
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


# Example usage (for testing purposes, remove in final)
if __name__ == "__main__":
    print("--- Configuration Check ---")
    print(f"DeepSeek API Key: {Config.DEEPSEEK_API_KEY}")
    print(f"MySQL Host: {Config.MYSQL_HOST}")
    print(f"Chroma DB Path: {Config.CHROMA_DB_PATH}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print(f"Reranker Model: {Config.RERANKER_MODEL_NAME}")
    print(f"Debug Mode: {Config.DEBUG_MODE}")
