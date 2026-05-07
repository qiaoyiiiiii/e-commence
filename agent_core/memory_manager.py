import json
import logging
from typing import Dict, Any, List

from database import Database
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db = Database()
        self.short_term_memory: List[Dict[str, str]] = [] # Stores current conversation history
        self.long_term_memory: Dict[str, Any] = {}
        self.load_long_term_memory()

    def load_long_term_memory(self):
        """ Loads user's long-term memory from the MySQL database. """
        query = "SELECT preferences, forbidden_items, chat_history FROM user_memory WHERE user_id = %s"
        result = self.db.execute_query(query, (self.user_id,), fetch_type='one')

        if result:
            self.long_term_memory['preferences'] = json.loads(result['preferences']) if result['preferences'] else {}
            self.long_term_memory['forbidden_items'] = json.loads(result['forbidden_items']) if result['forbidden_items'] else []
            # For now, chat_history from DB is not directly loaded into short_term_memory
            # as short_term_memory is for current session. It can be used for context later.
            logging.info(f"Loaded long-term memory for user {self.user_id}")
        else:
            logging.info(f"No existing long-term memory for user {self.user_id}, initializing new one.")
            self.long_term_memory = {
                'preferences': {},
                'forbidden_items': [],
                'chat_history': [] # This will store historical chat summaries/important points
            }
            self._save_long_term_memory_to_db() # Create an entry for new user

    def update_long_term_memory(self, key: str, value: Any):
        """ Updates a specific key in the user's long-term memory. """
        self.long_term_memory[key] = value
        logging.debug(f"Updated long-term memory for {key}: {value}")
        self._save_long_term_memory_to_db()

    def _save_long_term_memory_to_db(self):
        """ Saves the current long-term memory to the MySQL database. """
        query = (
            "INSERT INTO user_memory (user_id, preferences, forbidden_items, chat_history) "
            "VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE preferences = VALUES(preferences), "
            "forbidden_items = VALUES(forbidden_items), chat_history = VALUES(chat_history), "
            "last_active_at = CURRENT_TIMESTAMP"
        )
        try:
            self.db.execute_query(query, (
                self.user_id,
                json.dumps(self.long_term_memory.get('preferences', {})),
                json.dumps(self.long_term_memory.get('forbidden_items', [])),
                json.dumps(self.long_term_memory.get('chat_history', [])),
            ))
            logging.info(f"Long-term memory saved for user {self.user_id}")
        except Exception as e:
            logging.error(f"Error saving long-term memory for user {self.user_id}: {e}")

    def add_message_to_short_term_memory(self, role: str, message: str):
        """ Adds a message to the current session's short-term memory. """
        self.short_term_memory.append({"role": role, "content": message})
        # Optionally, implement memory compression/summarization for short_term_memory here

    def get_short_term_memory(self, limit: int = -1) -> List[Dict[str, str]]:
        """ Retrieves the short-term memory (conversation history). """
        return self.short_term_memory if limit == -1 else self.short_term_memory[-limit:]

    def get_long_term_memory(self) -> Dict[str, Any]:
        """ Retrieves the entire long-term memory. """
        return self.long_term_memory

    def get_user_preferences(self) -> Dict[str, Any]:
        """ Retrieves user preferences from long-term memory. """
        return self.long_term_memory.get('preferences', {})

    def get_forbidden_items(self) -> List[str]:
        """ Retrieves forbidden items from long-term memory. """
        return self.long_term_memory.get('forbidden_items', [])

    def get_history_summary(self) -> str:
        """ Returns the stored conversation summary (empty string if none). """
        summary = self.long_term_memory.get('chat_history', '')
        return summary if isinstance(summary, str) else ''

    def compress_short_term_memory(self, llm) -> None:
        """Summarizes old messages into long-term memory when short-term exceeds threshold.

        Keeps the MEMORY_COMPRESSION_KEEP_RECENT most recent messages verbatim so the
        LLM always has immediate context; the rest are merged into a rolling summary
        stored in long_term_memory['chat_history'].
        """
        messages = self.short_term_memory
        if len(messages) < Config.MEMORY_COMPRESSION_THRESHOLD:
            return

        to_summarize = messages[:-Config.MEMORY_COMPRESSION_KEEP_RECENT]
        self.short_term_memory = messages[-Config.MEMORY_COMPRESSION_KEEP_RECENT:]

        conversation_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in to_summarize
        )

        existing_summary = self.get_history_summary()
        if existing_summary:
            prompt = (
                f"原有摘要：{existing_summary}\n\n"
                f"新增对话：\n{conversation_text}\n\n"
                f"请合并以上内容，生成更新后的简短摘要，保留用户的主要需求、偏好和关键信息，不超过200字："
            )
        else:
            prompt = (
                f"请将以下对话历史总结成简短摘要，保留用户的主要需求、偏好和关键信息，不超过200字：\n\n"
                f"{conversation_text}\n\n摘要："
            )

        try:
            result = llm.invoke(prompt)
            summary = result.content if hasattr(result, 'content') else str(result)
            self.long_term_memory['chat_history'] = summary
            self._save_long_term_memory_to_db()
            logging.info(f"Short-term memory compressed for user {self.user_id}.")
        except Exception as e:
            logging.error(f"Failed to compress short-term memory for user {self.user_id}: {e}")
            # Restore messages so nothing is lost on failure
            self.short_term_memory = messages

    # Future: Implement methods for global knowledge memory (e.g., loading tag library)

# Example usage
if __name__ == "__main__":
    # Ensure MySQL is running and database/tables are set up
    test_user_id = "test_user_001"
    print(f"\n--- Testing MemoryManager for {test_user_id} ---")

    # Initialize for a new user or load existing
    memory_manager = MemoryManager(test_user_id)
    print(f"Initial long-term memory: {memory_manager.get_long_term_memory()}")
    print(f"Initial short-term memory: {memory_manager.get_short_term_memory()}")

    # Add to short-term memory
    memory_manager.add_message_to_short_term_memory("user", "我喜欢蓝色的衣服。")
    memory_manager.add_message_to_short_term_memory("agent", "好的，已为您记录。")
    print(f"\nShort-term memory after adding messages: {memory_manager.get_short_term_memory()}")

    # Update long-term preferences
    current_preferences = memory_manager.get_user_preferences()
    current_preferences['favorite_color'] = 'blue'
    memory_manager.update_long_term_memory('preferences', current_preferences)

    # Update forbidden items
    current_forbidden = memory_manager.get_forbidden_items()
    if '红色' not in current_forbidden:
        current_forbidden.append('红色')
    memory_manager.update_long_term_memory('forbidden_items', current_forbidden)

    # Re-load memory to confirm persistence (simulating new session)
    print(f"\nReloading memory for {test_user_id}...")
    new_memory_manager = MemoryManager(test_user_id)
    print(f"Reloaded long-term memory: {new_memory_manager.get_long_term_memory()}")
    print(f"Reloaded user preferences: {new_memory_manager.get_user_preferences()}")
    print(f"Reloaded forbidden items: {new_memory_manager.get_forbidden_items()}")

    # Test with a different user
    another_user_id = "test_user_002"
    print(f"\n--- Testing MemoryManager for {another_user_id} ---")
    another_memory_manager = MemoryManager(another_user_id)
    print(f"Initial long-term memory for new user: {another_memory_manager.get_long_term_memory()}")

    # Clean up test data (optional)
    # db_instance = Database()
    # db_instance.execute_query("DELETE FROM user_memory WHERE user_id = %s", (test_user_id,))
    # db_instance.execute_query("DELETE FROM user_memory WHERE user_id = %s", (another_user_id,))
    # print("Test user data cleaned up.")
