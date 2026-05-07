import argparse
import logging
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import Config
from database import Database
from agent_core.react_agent import DeepSeekLLM
from rag_module.hybrid_retriever import HybridRetrieverManager
from data.prompt_templates import RAG_PROMPT_TEMPLATE, format_chat_history
from agent_core.memory_manager import MemoryManager
from agent_core.mcp_manager import MCPManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_docs(docs):
    """ Formats documents for inclusion in the prompt context. """
    return "\n\n".join([
        f"[来源:{d.metadata.get('name', '未知')}]\n{d.page_content}" # Using 'name' from good_data as source
        for d in docs
    ])

def initialize_rag_chain():
    """ Initializes and returns the RAG chain components. """
    logger.info("Initializing RAG chain components...")

    # Initialize Database (ensures tables exist and connection is ready)
    db = Database() # Singleton instance
    if not db.connection or not db.connection.is_connected():
        logger.error("Database connection failed. Exiting.")
        return None, None, None

    # Initialize LLM
    try:
        deepseek_llm = DeepSeekLLM()
        llm = deepseek_llm.get_llm()
    except ValueError as e:
        logger.error(f"Failed to initialize DeepSeek LLM: {e}")
        return None, None, None

    # Initialize Hybrid Retriever
    hybrid_retriever_manager = HybridRetrieverManager()
    retriever = hybrid_retriever_manager.get_retriever()

    logger.info("RAG chain components initialized successfully.")
    return llm, retriever, RAG_PROMPT_TEMPLATE

def run_agent_cli(user_id: str = "default_user"):
    """ Runs the agent in command-line interface mode. """
    logger.info(f"Starting {Config.PROJECT_NAME} CLI for user: {user_id}...")

    # Initialize Memory Manager
    # memory_manager = MemoryManager(user_id) # MCPManager will handle memory
    # logger.info(f"Loaded long-term memory for user {user_id}: {memory_manager.get_long_term_memory()}")

    # Initialize MCPManager which internally handles MemoryManager and LLM
    mcp_manager = MCPManager(user_id)

    # Initialize RAG components separately if MCPManager needs them for tools
    llm, retriever, rag_prompt = initialize_rag_chain()
    if llm is None or retriever is None or rag_prompt is None:
        logger.error("Agent initialization failed. Please check logs for details.")
        return

    # Create a document combining chain
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("\n" + "=" * 50)
    print(f"欢迎使用 {Config.PROJECT_NAME}！")
    print("输入 'exit' 或 'quit' 退出。")
    print("=" * 50 + "\n")

    while True:
        user_input = input("您有什么需求？ ")
        if user_input.lower() in ['exit', 'quit']:
            print("感谢您的使用，再见！")
            break

        logger.info(f"User query: {user_input}")
        try:
            mcp_manager.memory_manager.add_message_to_short_term_memory("user", user_input)

            # Build chat history context: summary (if any) + recent verbatim messages
            history = mcp_manager.memory_manager.get_short_term_memory()[:-1]
            history_summary = mcp_manager.memory_manager.get_history_summary()
            recent_str = format_chat_history(history)
            if history_summary and recent_str:
                chat_history_block = f"[历史摘要] {history_summary}\n[最近对话]\n{recent_str}\n\n"
            elif history_summary:
                chat_history_block = f"[历史摘要] {history_summary}\n\n"
            elif recent_str:
                chat_history_block = f"对话历史：\n{recent_str}\n\n"
            else:
                chat_history_block = ""

            # Enrich the query with the user's long-term preferences and forbidden items
            memory_hint = mcp_manager.get_memory_context_hint()
            query_input = f"{user_input}。[{memory_hint}]" if memory_hint else user_input

            response = rag_chain.invoke({"input": query_input, "chat_history": chat_history_block})
            answer = response["answer"]
            context_docs = response["context"]

            # Store agent response, then compress memory if threshold is reached
            mcp_manager.memory_manager.add_message_to_short_term_memory("agent", answer)
            mcp_manager.memory_manager.compress_short_term_memory(llm)

            print("\n" + "-" * 50)
            print("Agent 回复:")
            print(answer)
            print("\n参考资料:")
            for doc in context_docs:
                source_name = doc.metadata.get('name', doc.metadata.get('goods_id', '未知')) # Prefer name, fallback to goods_id
                print(f"- {source_name}: {doc.page_content[:50]}...")
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"Error during RAG chain invocation: {e}")
            print("抱歉，Agent 在处理您的请求时遇到了问题。")

if __name__ == "__main__":
    # For now, we will directly run the CLI. Argument parsing can be added later if needed.
    parser = argparse.ArgumentParser(description=f"{Config.PROJECT_NAME} CLI")
    parser.add_argument("--user_id", type=str, default="default_user", help="User ID for memory management")
    args = parser.parse_args()

    run_agent_cli(args.user_id)
