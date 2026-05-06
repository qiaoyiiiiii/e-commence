import logging
from typing import List, Dict, Any

from config import Config
from agent_core.memory_manager import MemoryManager
from agent_core.react_agent import DeepSeekLLM # Assuming DeepSeekLLM is part of react_agent.py
from data.prompt_templates import format_chat_history

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class MCPManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_manager = MemoryManager(user_id)
        self.llm_service = DeepSeekLLM() # Initialize DeepSeek LLM
        self.llm = self.llm_service.get_llm()
        logging.info(f"MCPManager initialized for user {self.user_id}")

    def process_user_input(self, user_input: str) -> str:
        """ Processes user input, manages context, and generates a response. """
        self.memory_manager.add_message_to_short_term_memory("user", user_input)
        logging.info(f"User {self.user_id} input: {user_input}")

        # --- Context Management (Simplified for now) ---
        # Combine short-term memory (chat history) and relevant long-term memory
        chat_history = self.memory_manager.get_short_term_memory()
        # Placeholder for more advanced context: user preferences, forbidden items, etc.
        long_term_context = self.memory_manager.get_long_term_memory()

        # For a simple initial response, just use chat history as context
        formatted_history = format_chat_history(chat_history)
        prompt_context = f"当前的对话历史：\n{formatted_history}\n\n用户：{user_input}\nAgent："

        # --- Intent Detection & Action (Placeholder for ReAct/Skills integration) ---
        # In a full implementation, this is where ReAct would decide to call tools or generate a direct response.
        # For now, we'll just pass the context to the LLM for a direct response.
        try:
            # Use LLM to generate a response based on the context
            response_content = self.llm_service.invoke(prompt_context)
            self.memory_manager.add_message_to_short_term_memory("agent", response_content)
            logging.info(f"Agent {self.user_id} response: {response_content}")
            return response_content
        except Exception as e:
            logging.error(f"Error in MCPManager processing user input for {self.user_id}: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。"

    # Future methods:
    # def detect_missing_information(self, current_state) -> List[str]:
    #     """ Detects if critical information is missing for a recommendation. """
    #     pass
    #
    # def generate_active_question(self, missing_info: List[str]) -> str:
    #     """ Generates a question to elicit missing information from the user. """
    #     pass
    #
    # def resolve_coreference(self, user_input: str) -> str:
    #     """ Resolves pronouns and ambiguous references in user input. """
    #     pass

# Example usage
if __name__ == "__main__":
    # Ensure MySQL is running, tables are set up, and DEEPSEEK_API_KEY is in .env
    test_user_id = "test_user_mcp_001"
    print(f"\n--- Testing MCPManager for {test_user_id} ---")

    mcp_manager = MCPManager(test_user_id)

    # Simulate a conversation
    responses = []
    queries = [
        "我想要一件夏天的连衣裙",
        "颜色要是蓝色的",
        "价格不要太贵",
        "给我推荐一下"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        agent_response = mcp_manager.process_user_input(query)
        print(f"Agent: {agent_response}")
        responses.append(agent_response)

    print("\n--- Conversation History (Short-term) ---")
    for msg in mcp_manager.memory_manager.get_short_term_memory():
        print(f"{msg['role']}: {msg['content']}")

    # Clean up test data (optional)
    # db_instance = Database()
    # db_instance.execute_query("DELETE FROM user_memory WHERE user_id = %s", (test_user_id,))
    # print("Test user data cleaned up.")
