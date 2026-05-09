"""
模块：react_agent.py
职责：
    实现基于 ReAct（Reasoning + Acting）框架的智能体引擎，以及独立的 DeepSeek LLM 封装。

    包含两个核心类：
    1. ReActAgentEngine：
       - 集成 DeepSeek LLM、技能路由器（SkillRouter）和 RAG 混合检索器，
         构建完整的 ReAct 循环推理智能体。
       - Agent 按照"Thought → Action → Observation"循环推理，直到给出 Final Answer。
       - 所有已注册的 Skill 和 RAG 检索均以 LangChain Tool 的形式注入 Agent。

    2. DeepSeekLLM：
       - 对 ChatOpenAI（DeepSeek兼容接口）的轻量封装，用于不需要 ReAct 循环的简单 LLM 调用场景
         （例如 MCPManager 中的直接对话生成、记忆压缩摘要等）。
       - 与 ReActAgentEngine 解耦，避免循环导入。

    注意：langchain-deepseek 与 langchain 0.2.x 不兼容（要求 langchain-core>=0.3.34），
    因此改用 langchain-openai 的 ChatOpenAI 并指定 DeepSeek 的兼容接口地址，
    功能完全等价。

使用方式：
    # ReAct 智能体（适合复杂多工具推理场景）
    engine = ReActAgentEngine(user_id="user_001")
    reply = engine.run("我想找一款性价比高的通勤包")

    # 直接 LLM 调用（适合简单生成/摘要场景）
    llm = DeepSeekLLM()
    text = llm.invoke("请总结以下对话：...")
"""

import logging
from typing import List, Optional

from langchain.agents import create_tool_calling_agent, AgentExecutor # 修改导入：使用 create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI  # 替换 langchain_deepseek，使用 OpenAI 兼容接口

from config import Config
from tools.tool_loader import get_all_tools
from data.prompt_templates import REACT_SYSTEM_PROMPT

# DeepSeek 的 OpenAI 兼容接口地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class ReActAgentEngine:
    """
    基于 ReAct 框架的智能体引擎。

    将 DeepSeek LLM、Skill 工具集和 RAG 产品搜索工具组合为一个完整的
    LangChain AgentExecutor，实现多轮推理与工具调用。

    工具注册来源：
    1. SkillRouter 中注册的所有技能（如过滤、推荐、对比、自我反思等）。
    2. HybridRetrieverManager 提供的混合检索器，封装为 product_search 工具。

    ReAct 循环上限：
        max_iterations=5，防止 Agent 陷入无限循环。

    属性：
        user_id (str)                  : 当前用户标识，用于日志追踪。
        llm (ChatOpenAI)               : LangChain ChatOpenAI 实例（指向 DeepSeek 接口）。
        tools (List[Tool])             : 注册到 Agent 的工具列表。
        agent_executor (AgentExecutor) : 封装了 ReAct 循环的执行器。
    """

    def __init__(self, user_id: str = "default_user"):
        """
        初始化 ReActAgentEngine，依次完成 LLM、工具集和 Agent 执行器的创建。

        参数：
            user_id (str): 用户标识符，默认为 "default_user"。

        异常：
            若 DEEPSEEK_API_KEY 未配置或 LLM 初始化失败，则抛出异常，
            阻止 Agent 在无效状态下运行。
        """
        self.user_id = user_id
        # 按顺序初始化：LLM → 工具 → Agent 执行器
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools(user_id)
        self.agent_executor = self._create_agent_executor()
        logging.info(f"ReActAgentEngine initialized for user {self.user_id}")

    def _initialize_llm(self):
        """
        初始化 DeepSeek LLM 实例，供 ReAct Agent 进行推理使用。

        使用 langchain_openai.ChatOpenAI 并指定 DeepSeek 的 OpenAI 兼容接口地址，
        与原 ChatDeepSeek 行为完全等价，但无版本冲突问题。

        返回：
            ChatOpenAI: 已配置好的 LLM 实例（指向 DeepSeek 接口）。

        异常：
            ValueError : DEEPSEEK_API_KEY 未在 .env 文件中配置时抛出。
            Exception  : ChatOpenAI 初始化失败时向上层抛出原始异常。
        """
        # API Key 是必要条件，缺失时直接报错，避免后续请求失败难以定位
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")

        try:
            llm = ChatOpenAI(
                model=Config.DEEPSEEK_MODEL_NAME,      # 模型版本，如 deepseek-chat
                temperature=Config.LLM_TEMPERATURE,    # 生成温度，控制输出随机性
                api_key=Config.DEEPSEEK_API_KEY,       # DeepSeek API 密钥
                base_url=DEEPSEEK_BASE_URL,            # DeepSeek OpenAI 兼容接口地址
            )
            logging.info(f"DeepSeek LLM initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM: {e}")
            raise

    def _initialize_tools(self, user_id: str) -> List[Tool]:
        """
        委托 tools/tool_loader.py 完成所有工具的加载与注册。

        tools/ 目录是 skills/ 与 Agent 之间的适配层，负责：
          - 将 Agent 的字符串输入转换为技能函数所需的结构化参数
          - 将技能返回的结构化数据格式化为可读字符串
          - 为每个工具提供清晰的中文描述，帮助 Agent 正确选择工具

        参数：
            user_id (str): 当前用户 ID，由需要用户上下文的工具（个性化推荐）使用。

        返回：
            List[Tool]: 全部可用工具列表，直接传入 AgentExecutor。
        """
        return get_all_tools(user_id)

    def _create_agent_executor(self) -> AgentExecutor:
        """
        使用已初始化的 LLM 和工具集，构建 LangChain AgentExecutor。
        
        注意：由于 langchain 版本兼容性问题，改用 create_tool_calling_agent，
        它适用于支持 Tool Calling 接口的模型（如 DeepSeek）。
        """
        # 构造适用于 Tool Calling Agent 的 Prompt
        system_prompt = REACT_SYSTEM_PROMPT
        
        # 创建 ChatPromptTemplate
        # Tool Calling Agent 不需要显式的 agent_scratchpad，它由框架内部管理
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ])

        # 创建 Tool Calling Agent
        agent = create_tool_calling_agent(
            llm=self.llm, 
            tools=self.tools, 
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=Config.DEBUG_MODE,
            max_iterations=5,
            handle_parsing_errors=True,
        )
        return agent_executor

    def run(self, user_input: str, chat_history: str = "") -> str:
        """
        执行一轮 ReAct 推理，处理用户输入并返回最终答案。

        参数：
            user_input    (str): 用户输入的自然语言查询（可含偏好增强前缀）。
            chat_history  (str): 格式化后的对话历史字符串，注入 Prompt 的
                                 {chat_history} 槽位，帮助 Agent 理解上下文。
                                 默认为空字符串（首轮对话）。

        返回：
            str: Agent 的最终回复文本（Final Answer）。
                 若执行出错，返回中文友好提示字符串。
        """
        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            return response.get("output", "抱歉，我无法生成回答。")
        except Exception as e:
            logging.error(f"Error running ReAct Agent: {e}")
            return f"抱歉，处理您的请求时出现问题：{str(e)}"


class DeepSeekLLM:
    """
    DeepSeek 大模型的轻量封装，提供直接调用接口（不经过 ReAct 循环）。

    适用场景：
    - MCPManager 中的简单对话生成（直接将提示词发给 LLM 获取回复）。
    - MemoryManager 中的历史摘要压缩（调用 compress_short_term_memory 时传入此实例）。
    - 其他不需要工具调用和多步推理的简单文本生成任务。

    与 ReActAgentEngine 的区别：
    - 不注册任何工具，不执行 Thought/Action 循环。
    - 初始化更轻量，仅创建 ChatOpenAI 实例。
    - 与 ReActAgentEngine 解耦，避免在 mcp_manager.py 中导入时产生循环依赖。

    属性：
        llm (ChatOpenAI): 底层 LangChain ChatOpenAI 实例（指向 DeepSeek 接口）。
    """

    def __init__(self):
        """
        初始化 DeepSeekLLM，创建底层 ChatOpenAI 实例。

        异常：
            ValueError : DEEPSEEK_API_KEY 未配置时抛出。
            Exception  : ChatOpenAI 初始化失败时向上层抛出原始异常。
        """
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        初始化 ChatOpenAI 实例（指向 DeepSeek 兼容接口），配置项与 ReActAgentEngine 中相同。

        返回：
            ChatOpenAI: 已配置好的 LLM 实例。

        异常：
            ValueError : DEEPSEEK_API_KEY 未配置时抛出。
            Exception  : 初始化失败时向上层抛出。
        """
        # API Key 缺失时立即报错，避免后续请求失败难以定位
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")

        try:
            llm = ChatOpenAI(
                model=Config.DEEPSEEK_MODEL_NAME,      # 模型版本
                temperature=Config.LLM_TEMPERATURE,    # 生成温度
                api_key=Config.DEEPSEEK_API_KEY,       # API 密钥
                base_url=DEEPSEEK_BASE_URL,            # DeepSeek OpenAI 兼容接口地址
            )
            logging.info(f"DeepSeek LLM (Direct) initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM (Direct): {e}")
            raise

    def get_llm(self):
        """
        获取底层 ChatOpenAI 实例，供需要直接操作 LangChain LLM 的场景使用。

        返回：
            ChatOpenAI: 底层 LLM 实例。
        """
        return self.llm

    def invoke(self, prompt: str):
        """
        直接将提示词发送给 DeepSeek LLM 并返回生成文本（不经过 ReAct 循环）。

        参数：
            prompt (str): 完整的提示词字符串，包含上下文和指令。

        返回：
            str: LLM 生成的文本内容。
                 若调用失败，返回空字符串 ""，并记录错误日志。

        注意：
            返回值为纯文本字符串（已从 LangChain AIMessage 中提取 .content），
            调用方无需再处理 Message 对象。
        """
        try:
            response = self.llm.invoke(prompt)
            # LangChain ChatOpenAI 返回 AIMessage 对象，提取 .content 得到文本
            return response.content
        except Exception as e:
            logging.error(f"Error invoking DeepSeek LLM directly: {e}")
            # 返回空字符串而非抛出异常，防止上层调用因 LLM 故障崩溃
            return ""


# Example usage
if __name__ == "__main__":
    # Make sure to set DEEPSEEK_API_KEY in your .env file before running this example
    try:
        agent_engine = ReActAgentEngine(user_id="test_user")

        test_query = "我想找一个适合通勤的包包，价格不要太贵，最好有性价比。"
        print(f"\nUser Query: {test_query}")
        print("-" * 50)

        response = agent_engine.run(test_query)
        print(f"Agent Response:\n{response}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")