"""
模块：memory_manager.py
职责：
    管理电商导购智能体的用户记忆系统，分为两个层次：
    - 短期记忆（short_term_memory）：当前会话的对话消息列表，仅在本次运行中存在。
    - 长期记忆（long_term_memory）：用户偏好、禁止商品、历史对话摘要，持久化存储在 MySQL 数据库中。

依赖：
    - database.Database       : 封装了 MySQL 连接与查询的数据库工具类
    - config.Config           : 全局配置（日志级别、压缩阈值等）
    - json                    : 用于将 Python 对象与 JSON 字符串互相转换后存入数据库
    - logging                 : 标准日志记录

使用方式：
    memory = MemoryManager(user_id="user_001")
    memory.add_message_to_short_term_memory("user", "我想买一件蓝色连衣裙")
    memory.update_long_term_memory("preferences", {"color": "blue"})
    memory.compress_short_term_memory(llm)   # 当短期记忆过长时压缩为摘要
"""

import json
import logging
from typing import Dict, Any, List

from database import Database
from config import Config

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class MemoryManager:
    """
    用户记忆管理器。

    负责维护和持久化用户的两类记忆：
    - 短期记忆：当前会话中按时间顺序排列的对话消息（role + content 字典列表）。
    - 长期记忆：跨会话保留的用户偏好（preferences）、禁止推荐商品（forbidden_items）
      以及历史对话摘要（chat_history），存储在数据库表 user_memory 中。

    属性：
        user_id (str)              : 当前用户的唯一标识符。
        db (Database)              : 数据库连接实例。
        short_term_memory (list)   : 当前会话消息列表，元素格式为 {"role": ..., "content": ...}。
        long_term_memory (dict)    : 从数据库加载的长期记忆字典。
    """

    def __init__(self, user_id: str):
        """
        初始化 MemoryManager，并立即从数据库加载该用户的长期记忆。

        参数：
            user_id (str): 用户唯一标识符，用于数据库查询与写入。
        """
        self.user_id = user_id
        self.db = Database()
        # 短期记忆：仅在当前会话中存在，存储对话消息历史
        self.short_term_memory: List[Dict[str, str]] = []
        # 长期记忆：从数据库加载，包含偏好、禁止项、历史摘要
        self.long_term_memory: Dict[str, Any] = {}
        # 初始化时立即从数据库加载长期记忆
        self.load_long_term_memory()

    def load_long_term_memory(self):
        """
        从 MySQL 数据库的 user_memory 表加载当前用户的长期记忆。

        行为：
            - 若数据库中存在该用户记录，则将 preferences 和 forbidden_items 字段
              从 JSON 字符串反序列化后存入 self.long_term_memory。
            - 若不存在，则初始化空的长期记忆结构，并立即写入数据库（为新用户建档）。

        注意：
            chat_history 字段从数据库读取后暂不写入短期记忆，
            仅在需要时通过 get_history_summary() 提取作为上下文提示使用。
        """
        query = "SELECT preferences, forbidden_items, chat_history FROM user_memory WHERE user_id = %s"
        result = self.db.execute_query(query, (self.user_id,), fetch_type='one')

        if result:
            # 将数据库中存储的 JSON 字符串反序列化为 Python 对象
            self.long_term_memory['preferences'] = (
                json.loads(result['preferences']) if result['preferences'] else {}
            )
            self.long_term_memory['forbidden_items'] = (
                json.loads(result['forbidden_items']) if result['forbidden_items'] else []
            )
            # chat_history 字段不直接加载到短期记忆，短期记忆仅用于当前会话
            logging.info(f"Loaded long-term memory for user {self.user_id}")
        else:
            # 新用户：初始化空结构并写入数据库，为后续更新建立记录
            logging.info(f"No existing long-term memory for user {self.user_id}, initializing new one.")
            self.long_term_memory = {
                'preferences': {},        # 用户偏好键值对
                'forbidden_items': [],    # 不希望被推荐的商品列表
                'chat_history': []        # 历史对话的摘要（后续会压缩为字符串）
            }
            self._save_long_term_memory_to_db()  # 为新用户在数据库中创建初始记录

    def update_long_term_memory(self, key: str, value: Any):
        """
        更新长期记忆中指定键的值，并立即同步到数据库。

        参数：
            key (str) : 要更新的字段名，如 'preferences'、'forbidden_items'、'chat_history'。
            value (Any): 对应的新值，通常为 dict 或 list。
        """
        self.long_term_memory[key] = value
        logging.debug(f"Updated long-term memory for {key}: {value}")
        # 每次更新后立即持久化，确保数据不丢失
        self._save_long_term_memory_to_db()

    def _save_long_term_memory_to_db(self):
        """
        将当前 long_term_memory 的内容序列化后写入（或更新）数据库。

        SQL 策略：
            使用 INSERT ... ON DUPLICATE KEY UPDATE，以 user_id 为唯一键：
            - 若该用户已存在记录，则更新 preferences、forbidden_items、chat_history
              以及 last_active_at 时间戳。
            - 若不存在，则插入新行。

        异常处理：
            捕获所有异常并记录错误日志，不向上层抛出，避免记忆写入失败影响主流程。
        """
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
                # 将 Python 对象序列化为 JSON 字符串存入数据库文本字段
                json.dumps(self.long_term_memory.get('preferences', {})),
                json.dumps(self.long_term_memory.get('forbidden_items', [])),
                json.dumps(self.long_term_memory.get('chat_history', [])),
            ))
            logging.info(f"Long-term memory saved for user {self.user_id}")
        except Exception as e:
            logging.error(f"Error saving long-term memory for user {self.user_id}: {e}")

    def add_message_to_short_term_memory(self, role: str, message: str):
        """
        向当前会话的短期记忆中追加一条消息。

        参数：
            role (str)   : 消息角色，通常为 'user' 或 'agent'。
            message (str): 消息的文本内容。

        注意：
            短期记忆仅在进程内存中维护，会话结束后会丢失。
            若需持久化，需调用 compress_short_term_memory() 压缩后存入长期记忆。
        """
        self.short_term_memory.append({"role": role, "content": message})

    def get_short_term_memory(self, limit: int = -1) -> List[Dict[str, str]]:
        """
        获取当前会话的短期记忆（对话历史）。

        参数：
            limit (int): 返回最近 limit 条消息。默认 -1 表示返回全部消息。

        返回：
            List[Dict[str, str]]: 消息列表，每条格式为 {"role": ..., "content": ...}。
        """
        # limit 为 -1 时返回完整历史，否则只返回最后 limit 条
        return self.short_term_memory if limit == -1 else self.short_term_memory[-limit:]

    def get_long_term_memory(self) -> Dict[str, Any]:
        """
        获取完整的长期记忆字典。

        返回：
            Dict[str, Any]: 包含 preferences、forbidden_items、chat_history 等字段的字典。
        """
        return self.long_term_memory

    def get_user_preferences(self) -> Dict[str, Any]:
        """
        获取用户偏好设置。

        返回：
            Dict[str, Any]: 偏好键值对，例如 {"color": "blue", "style": "casual"}。
                            若未设置则返回空字典。
        """
        return self.long_term_memory.get('preferences', {})

    def get_forbidden_items(self) -> List[str]:
        """
        获取用户不希望被推荐的商品或属性列表。

        返回：
            List[str]: 禁止项列表，例如 ["红色", "皮草"]。
                       若未设置则返回空列表。
        """
        return self.long_term_memory.get('forbidden_items', [])

    def get_history_summary(self) -> str:
        """
        获取历史对话摘要字符串。

        返回：
            str: 由 compress_short_term_memory() 生成的对话摘要。
                 若尚无摘要（初始状态或摘要为非字符串类型），则返回空字符串。

        注意：
            chat_history 字段初始化时可能为空列表 []，此时返回空字符串，
            防止将列表直接传入提示词导致格式错误。
        """
        summary = self.long_term_memory.get('chat_history', '')
        # 仅当 chat_history 为字符串时才作为摘要返回，避免初始空列表被误用
        return summary if isinstance(summary, str) else ''

    def compress_short_term_memory(self, llm) -> None:
        """
        当短期记忆消息数量超过阈值时，调用 LLM 将较早的对话压缩为文字摘要，
        并将摘要滚动合并到长期记忆的 chat_history 字段中。

        压缩策略：
            1. 保留最近 MEMORY_COMPRESSION_KEEP_RECENT 条消息不压缩，维持即时上下文。
            2. 将更早的消息拼接成文本，构造提示词让 LLM 生成摘要。
            3. 若已有历史摘要，则要求 LLM 合并新旧摘要，实现"滚动摘要"。
            4. 将生成的摘要写入 long_term_memory['chat_history'] 并持久化到数据库。

        参数：
            llm: 支持 .invoke(prompt) 接口的 LangChain LLM 对象，
                 返回值需有 .content 属性或可转为字符串。

        异常处理：
            若 LLM 调用失败，恢复原始短期记忆，确保消息不丢失，并记录错误日志。
        """
        messages = self.short_term_memory

        # 若消息数量未达到压缩阈值，直接返回，无需操作
        if len(messages) < Config.MEMORY_COMPRESSION_THRESHOLD:
            return

        # 切分：需要压缩的旧消息 + 保留的新消息
        to_summarize = messages[:-Config.MEMORY_COMPRESSION_KEEP_RECENT]
        self.short_term_memory = messages[-Config.MEMORY_COMPRESSION_KEEP_RECENT:]

        # 将待压缩消息格式化为可读的对话文本，供 LLM 理解
        conversation_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in to_summarize
        )

        # 根据是否已有历史摘要，构造不同的提示词
        existing_summary = self.get_history_summary()
        if existing_summary:
            # 已有摘要：要求 LLM 合并原摘要与新对话，实现滚动更新
            prompt = (
                f"原有摘要：{existing_summary}\n\n"
                f"新增对话：\n{conversation_text}\n\n"
                f"请合并以上内容，生成更新后的简短摘要，保留用户的主要需求、偏好和关键信息，不超过200字："
            )
        else:
            # 首次压缩：直接对当前对话进行总结
            prompt = (
                f"请将以下对话历史总结成简短摘要，保留用户的主要需求、偏好和关键信息，不超过200字：\n\n"
                f"{conversation_text}\n\n摘要："
            )

        try:
            result = llm.invoke(prompt)
            # 兼容 LangChain Message 对象（有 .content 属性）和普通字符串返回值
            summary = result.content if hasattr(result, 'content') else str(result)
            # 将新摘要写入长期记忆并持久化到数据库
            self.long_term_memory['chat_history'] = summary
            self._save_long_term_memory_to_db()
            logging.info(f"Short-term memory compressed for user {self.user_id}.")
        except Exception as e:
            logging.error(f"Failed to compress short-term memory for user {self.user_id}: {e}")
            # LLM 调用失败时恢复完整的短期记忆，防止数据丢失
            self.short_term_memory = messages


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
