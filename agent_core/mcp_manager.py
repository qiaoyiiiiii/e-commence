"""
模块：mcp_manager.py
职责：
    MCP（Multi-turn Conversation Processing，多轮对话处理）管理器。
    作为用户会话的核心协调层，负责：
    1. 通过 MemoryManager 维护用户的短期与长期记忆。
    2. 将用户输入结合对话历史构造上下文，调用 DeepSeekLLM 生成回复。
    3. 提供 get_memory_context_hint() 接口，将用户偏好/禁止项拼接为查询增强字符串，
       供 RAG 检索模块使用。
    4. 对 LLM 服务采用懒加载（Lazy Initialization）策略，仅在实际调用时才初始化，
       节省无需 LLM 时的冷启动开销。

依赖：
    - config.Config                        : 全局配置（日志级别等）
    - agent_core.memory_manager.MemoryManager : 用户记忆管理
    - agent_core.react_agent.DeepSeekLLM   : DeepSeek 大模型封装（直接调用模式）
    - data.prompt_templates.format_chat_history : 将消息列表格式化为对话文本的工具函数

使用方式：
    manager = MCPManager(user_id="user_001")
    hint = manager.get_memory_context_hint()    # 获取偏好提示字符串，用于 RAG 查询增强
    reply = manager.process_user_input("我想买一件连衣裙")  # 处理用户输入并返回回复
"""

import logging
from typing import List, Dict, Any

from config import Config
from agent_core.memory_manager import MemoryManager
from agent_core.react_agent import DeepSeekLLM  # DeepSeekLLM 定义于 react_agent.py
from data.prompt_templates import format_chat_history

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class MCPManager:
    """
    多轮对话处理管理器（MCP Manager）。

    该类是用户与智能体之间的会话协调中心，集成记忆管理与 LLM 调用功能。
    主要职责：
    - 初始化时加载用户历史记忆（通过 MemoryManager）。
    - 每轮对话将用户输入写入短期记忆，并将对话历史作为上下文传给 LLM。
    - LLM 服务（DeepSeekLLM）采用懒加载，只有在实际需要生成回复时才初始化，
      避免在仅需记忆查询等轻量操作时浪费资源。
    - 提供 get_memory_context_hint() 方法，将用户偏好和禁止项组合为查询附加字符串，
      可直接拼接到 RAG 检索的查询中以提升个性化效果。

    属性：
        user_id (str)              : 当前用户唯一标识符。
        memory_manager (MemoryManager): 负责读写用户短期/长期记忆的实例。
        _llm_service (DeepSeekLLM | None): LLM 服务实例，懒加载，初始为 None。
    """

    def __init__(self, user_id: str):
        """
        初始化 MCPManager，加载用户记忆，但不立即初始化 LLM。

        参数：
            user_id (str): 用户唯一标识符，用于关联记忆数据库记录。
        """
        self.user_id = user_id
        # 实例化 MemoryManager，触发从数据库加载该用户的历史长期记忆
        self.memory_manager = MemoryManager(user_id)
        self._llm_service = None
        # ReActAgentEngine 懒加载：仅在首次调用 react_engine 属性时初始化，
        # 避免导入 MCPManager 时触发向量库、BM25、LLM 等重型组件的加载
        self._react_engine = None
        logging.info(f"MCPManager initialized for user {self.user_id}")

    @property
    def react_engine(self):
        """
        ReActAgentEngine 的懒加载属性。

        首次访问时才创建 ReActAgentEngine 实例（触发 SkillRouter、
        HybridRetrieverManager、LLM 等重型组件的初始化），后续调用直接返回。
        延迟初始化确保仅在真正需要处理用户输入时才承担启动开销。

        返回：
            ReActAgentEngine: 已初始化的 ReAct 智能体引擎实例。
        """
        if self._react_engine is None:
            # 延迟导入，避免与 react_agent.py 之间的潜在循环依赖
            from agent_core.react_agent import ReActAgentEngine
            self._react_engine = ReActAgentEngine(user_id=self.user_id)
        return self._react_engine

    @property
    def llm_service(self):
        """
        LLM 服务的懒加载属性。

        首次访问时才实例化 DeepSeekLLM，后续调用直接返回已有实例。
        这样可以在仅需记忆读写操作时（如 get_memory_context_hint）
        避免不必要的 LLM 初始化开销。

        返回：
            DeepSeekLLM: 已初始化的 DeepSeek 大模型服务实例。
        """
        if self._llm_service is None:
            # 第一次访问时才创建 LLM 实例（懒加载）
            self._llm_service = DeepSeekLLM()
        return self._llm_service

    def get_memory_context_hint(self) -> str:
        """
        生成用户记忆上下文提示字符串，用于在 RAG 检索前增强查询语义。

        从长期记忆中读取用户偏好和禁止项，将其格式化为自然语言片段，
        调用方可将返回值拼接到原始查询字符串中，使检索结果更贴合用户个性化需求。

        返回：
            str: 格式化的提示字符串，例如：
                 "用户偏好(color:blue, style:casual)；不要推荐(红色, 皮草)"
                 若偏好和禁止项均为空，则返回空字符串 ""。

        示例：
            hint = manager.get_memory_context_hint()
            enriched_query = f"{hint} {user_query}" if hint else user_query
        """
        preferences = self.memory_manager.get_user_preferences()
        forbidden_items = self.memory_manager.get_forbidden_items()
        hints = []

        # 若存在用户偏好，将所有键值对拼接为"key:value"形式
        if preferences:
            pref_parts = [f"{k}:{v}" for k, v in preferences.items()]
            hints.append(f"用户偏好({', '.join(pref_parts)})")

        # 若存在禁止项，列出所有不希望推荐的商品或属性
        if forbidden_items:
            hints.append(f"不要推荐({', '.join(forbidden_items)})")

        # 用中文分号拼接多个提示片段
        return '；'.join(hints)

    def process_user_input(self, user_input: str) -> str:
        """
        处理一轮用户输入的完整流程：记忆更新 → 历史构建 → 偏好增强 → Agent 推理 → 记忆压缩。

        这是整个对话系统的核心调度方法，调用方（main.py）只需传入用户原始输入，
        其余所有步骤（记忆、检索、工具调用、LLM 推理）均在此方法内部协调完成。

        流程：
            1. 将用户消息写入短期记忆。
            2. 构建对话历史上下文块（摘要 + 近期原始消息）。
            3. 将用户长期偏好拼入查询，增强检索个性化效果。
            4. 调用 ReActAgentEngine.run()，由 Agent 自主决策工具并生成答案。
            5. 将 Agent 回复写入短期记忆，按需触发 LLM 摘要压缩。

        参数：
            user_input (str): 用户本轮输入的原始文本。

        返回：
            str: ReAct Agent 生成的最终回复文本。
                 若 Agent 执行出错，返回中文友好提示字符串。
        """
        # 步骤1：记录用户消息
        self.memory_manager.add_message_to_short_term_memory("user", user_input)
        logging.info(f"User [{self.user_id}] input: {user_input}")

        # 步骤2：构建对话历史上下文（与 main.py 原有逻辑一致）
        history = self.memory_manager.get_short_term_memory()[:-1]  # 排除刚写入的当前消息
        history_summary = self.memory_manager.get_history_summary()
        recent_str = format_chat_history(history)

        if history_summary and recent_str:
            chat_history_block = f"[历史摘要] {history_summary}\n[最近对话]\n{recent_str}\n\n"
        elif history_summary:
            chat_history_block = f"[历史摘要] {history_summary}\n\n"
        elif recent_str:
            chat_history_block = f"对话历史：\n{recent_str}\n\n"
        else:
            chat_history_block = ""

        # 步骤3：将用户长期偏好拼入查询，引导 Agent 和检索器关注个性化需求
        memory_hint = self.get_memory_context_hint()
        enriched_input = f"{user_input}。[{memory_hint}]" if memory_hint else user_input

        # 步骤4：ReAct Agent 自主决策——选择工具、调用技能、生成最终答案
        response = self.react_engine.run(enriched_input, chat_history=chat_history_block)
        logging.info(f"Agent [{self.user_id}] response: {response[:100]}...")

        # 步骤5：记录回复并按需压缩历史摘要
        self.memory_manager.add_message_to_short_term_memory("agent", response)
        # 复用 react_engine 内已有的 LLM 实例，避免创建重复对象
        self.memory_manager.compress_short_term_memory(self.react_engine.llm)

        return response

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
# if __name__ == "__main__":
#     # Ensure MySQL is running, tables are set up, and DEEPSEEK_API_KEY is in .env
#     test_user_id = "test_user_mcp_001"
#     print(f"\n--- Testing MCPManager for {test_user_id} ---")

#     mcp_manager = MCPManager(test_user_id)

#     # Simulate a conversation
#     responses = []
#     queries = [
#         "我想要一件夏天的连衣裙",
#         "颜色要是蓝色的",
#         "价格不要太贵",
#         "给我推荐一下"
#     ]

#     for query in queries:
#         print(f"\nUser: {query}")
#         agent_response = mcp_manager.process_user_input(query)
#         print(f"Agent: {agent_response}")
#         responses.append(agent_response)

#     print("\n--- Conversation History (Short-term) ---")
#     for msg in mcp_manager.memory_manager.get_short_term_memory():
#         print(f"{msg['role']}: {msg['content']}")

    # Clean up test data (optional)
    # db_instance = Database()
    # db_instance.execute_query("DELETE FROM user_memory WHERE user_id = %s", (test_user_id,))
    # print("Test user data cleaned up.")
