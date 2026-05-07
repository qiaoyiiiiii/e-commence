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
       - 对 ChatDeepSeek 的轻量封装，用于不需要 ReAct 循环的简单 LLM 调用场景
         （例如 MCPManager 中的直接对话生成、记忆压缩摘要等）。
       - 与 ReActAgentEngine 解耦，避免循环导入。

依赖：
    - langchain.agents              : create_react_agent、AgentExecutor
    - langchain_core.prompts        : PromptTemplate
    - langchain_core.tools          : Tool
    - langchain_deepseek            : ChatDeepSeek（DeepSeek 官方 LangChain 集成）
    - config.Config                 : API Key、模型名称、温度等配置项
    - rag_module.hybrid_retriever   : HybridRetrieverManager（向量+关键词混合检索）
    - data.prompt_templates         : REACT_AGENT_PROMPT（备用，当前使用模块内定义的提示词）
    - agent_core.skill_router       : SkillRouter（在方法内延迟导入，避免循环依赖）

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

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_deepseek import ChatDeepSeek

from config import Config
from rag_module.hybrid_retriever import HybridRetrieverManager
from data.prompt_templates import REACT_AGENT_PROMPT

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# ReAct 提示词模板（标准 ReAct 格式）
# 占位符说明：
#   {tools}           - LangChain 自动填充已注册工具的名称和描述
#   {tool_names}      - LangChain 自动填充工具名称列表（逗号分隔）
#   {input}           - 用户本轮输入的问题
#   {agent_scratchpad}- LangChain 自动填充 Agent 的中间推理步骤记录
REACT_SYSTEM_PROMPT = """
你是一个智能电商导购助手。你的目标是帮助用户找到合适的商品。
你可以使用以下工具：

{tools}

使用以下格式进行思考和行动：

Question: 用户输入的问题
Thought: 你应该始终思考下一步该做什么
Action: 应该采取的行动，必须是 [{tool_names}] 中的一个
Action Input: 行动的输入参数
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终回答

开始！

Question: {input}
Thought:{agent_scratchpad}
"""


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
        llm (ChatDeepSeek)             : LangChain DeepSeek LLM 实例。
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
        self.tools = self._initialize_tools()
        self.agent_executor = self._create_agent_executor()
        logging.info(f"ReActAgentEngine initialized for user {self.user_id}")

    def _initialize_llm(self):
        """
        初始化 DeepSeek LLM 实例，供 ReAct Agent 进行推理使用。

        从 Config 读取 API Key、模型名称和温度参数，通过 langchain_deepseek
        的 ChatDeepSeek 接口创建 LLM 实例。

        返回：
            ChatDeepSeek: 已配置好的 LLM 实例。

        异常：
            ValueError : DEEPSEEK_API_KEY 未在 .env 文件中配置时抛出。
            Exception  : ChatDeepSeek 初始化失败时向上层抛出原始异常。
        """
        # API Key 是必要条件，缺失时直接报错，避免后续请求失败难以定位
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")

        try:
            llm = ChatDeepSeek(
                model=Config.DEEPSEEK_MODEL_NAME,      # 模型版本，如 deepseek-chat
                temperature=Config.LLM_TEMPERATURE,    # 生成温度，控制输出随机性
                api_key=Config.DEEPSEEK_API_KEY,       # DeepSeek API 密钥
            )
            logging.info(f"DeepSeek LLM initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM: {e}")
            raise

    def _initialize_tools(self) -> List[Tool]:
        """
        初始化 Agent 可用的工具列表，来源为 SkillRouter 和 RAG 检索器。

        工具注册流程：
            1. 延迟导入 SkillRouter（避免与 skill_router.py 产生循环依赖）。
            2. 遍历 SkillRouter 中已注册的所有技能，每个技能封装为一个 LangChain Tool。
               - Tool 的 func 为闭包，固定捕获技能名称（name=skill_name），
                 避免 Python 循环变量绑定的陷阱。
               - 技能调用接口：skill_router.execute_skill(name, user_query=query)。
            3. 尝试初始化 HybridRetrieverManager，将混合检索器封装为 product_search 工具。
               - 若 RAG 初始化失败（如向量库未建立），仅记录警告，不影响其他工具。

        返回：
            List[Tool]: 可注入到 AgentExecutor 的工具列表。

        注意：
            RAG 工具初始化失败时不抛出异常，Agent 仍可使用其他技能工具正常运行。
        """
        # 延迟导入，避免 react_agent ↔ skill_router 的循环导入问题
        from agent_core.skill_router import SkillRouter

        tools = []

        # 步骤 1：将 SkillRouter 中所有已注册技能封装为 LangChain Tool
        skill_router = SkillRouter()
        registered_skills = skill_router.list_skills()

        for skill_name in registered_skills:
            # 使用默认参数 name=skill_name 固定闭包中的技能名称，
            # 防止 Python 循环中所有闭包都引用最后一个 skill_name 的问题
            def skill_wrapper(query, name=skill_name):
                try:
                    return skill_router.execute_skill(name, user_query=query)
                except Exception as e:
                    return f"Error executing skill {name}: {str(e)}"

            tool = Tool(
                name=skill_name,
                func=skill_wrapper,
                description=f"Useful for when you need to {skill_name}. Input should be a relevant query or parameters."
            )
            tools.append(tool)

        # 步骤 2：将 RAG 混合检索器封装为 product_search 工具
        try:
            hybrid_retriever_manager = HybridRetrieverManager()
            retriever = hybrid_retriever_manager.get_retriever()

            def rag_search(query: str) -> str:
                """
                基于用户自然语言描述，通过混合检索器搜索相关商品。

                参数：
                    query (str): 用户的商品需求描述。

                返回：
                    str: 格式化的商品信息列表，每行包含名称、价格和特性；
                         若无匹配结果则返回提示字符串。
                """
                docs = retriever.invoke(query)
                if not docs:
                    return "No relevant products found."

                formatted_docs = []
                for doc in docs:
                    # 从文档元数据中提取商品信息，缺失字段用默认值填充
                    name = doc.metadata.get('name', 'Unknown')
                    price = doc.metadata.get('price', 'N/A')
                    feature = doc.metadata.get('feature', '')
                    formatted_docs.append(f"Product: {name}, Price: {price}, Feature: {feature}")

                return "\n".join(formatted_docs)

            rag_tool = Tool(
                name="product_search",
                func=rag_search,
                description="Useful for when you need to search for products based on natural language descriptions. Input should be a detailed product description or requirement."
            )
            tools.append(rag_tool)
        except Exception as e:
            # RAG 工具不可用时仅发出警告，Agent 仍可运行其他 Skill 工具
            logging.warning(f"Failed to initialize RAG tool: {e}. RAG functionality will be unavailable.")

        return tools

    def _create_agent_executor(self) -> AgentExecutor:
        """
        使用已初始化的 LLM 和工具集，构建 LangChain ReAct AgentExecutor。

        流程：
            1. 基于 REACT_SYSTEM_PROMPT 创建 PromptTemplate。
            2. 调用 create_react_agent 将 LLM、工具和提示词组合为 ReAct Agent。
            3. 将 Agent 包装为 AgentExecutor，配置最大迭代次数和错误处理策略。

        返回：
            AgentExecutor: 可直接调用 .invoke() 执行 ReAct 推理的执行器对象。

        配置说明：
            verbose=Config.DEBUG_MODE   : 调试模式下打印每步推理过程。
            max_iterations=5            : 最多执行 5 轮 Thought/Action/Observation 循环，
                                          防止因工具异常或 LLM 误判导致的无限循环。
            handle_parsing_errors=True  : LLM 输出格式不符合 ReAct 要求时自动重试，
                                          提高 Agent 鲁棒性。
        """
        # 从字符串模板创建 PromptTemplate，LangChain 会自动填充 {tools}、{tool_names} 等占位符
        prompt = PromptTemplate.from_template(REACT_SYSTEM_PROMPT)

        # 组合 LLM、工具和提示词，生成标准 ReAct Agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # 将 Agent 包装为执行器，添加循环控制和错误处理
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=Config.DEBUG_MODE,   # DEBUG 模式下输出详细推理步骤
            max_iterations=5,            # 防止无限循环
            handle_parsing_errors=True   # 自动处理 LLM 输出解析错误
        )
        return agent_executor

    def run(self, user_input: str) -> str:
        """
        执行一轮 ReAct 推理，处理用户输入并返回最终答案。

        Agent 会在内部进行多轮 Thought/Action/Observation 循环，
        直到得出 Final Answer 或达到最大迭代次数。

        参数：
            user_input (str): 用户输入的自然语言查询。

        返回：
            str: Agent 的最终回复文本。
                 若执行出错，返回包含错误信息的友好提示字符串。
        """
        try:
            # AgentExecutor.invoke 接收字典格式的输入，"input" 对应提示词中的 {input} 占位符
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "Sorry, I couldn't generate a response.")
        except Exception as e:
            logging.error(f"Error running ReAct Agent: {e}")
            return f"Sorry, an error occurred while processing your request: {str(e)}"


class DeepSeekLLM:
    """
    DeepSeek 大模型的轻量封装，提供直接调用接口（不经过 ReAct 循环）。

    适用场景：
    - MCPManager 中的简单对话生成（直接将提示词发给 LLM 获取回复）。
    - MemoryManager 中的历史摘要压缩（调用 compress_short_term_memory 时传入此实例）。
    - 其他不需要工具调用和多步推理的简单文本生成任务。

    与 ReActAgentEngine 的区别：
    - 不注册任何工具，不执行 Thought/Action 循环。
    - 初始化更轻量，仅创建 ChatDeepSeek 实例。
    - 与 ReActAgentEngine 解耦，避免在 mcp_manager.py 中导入时产生循环依赖。

    属性：
        llm (ChatDeepSeek): 底层 LangChain DeepSeek LLM 实例。
    """

    def __init__(self):
        """
        初始化 DeepSeekLLM，创建底层 ChatDeepSeek 实例。

        异常：
            ValueError : DEEPSEEK_API_KEY 未配置时抛出。
            Exception  : ChatDeepSeek 初始化失败时向上层抛出原始异常。
        """
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        初始化 ChatDeepSeek 实例，配置项与 ReActAgentEngine 中相同。

        返回：
            ChatDeepSeek: 已配置好的 LLM 实例。

        异常：
            ValueError : DEEPSEEK_API_KEY 未配置时抛出。
            Exception  : 初始化失败时向上层抛出。
        """
        # API Key 缺失时立即报错，避免后续请求失败难以定位
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")

        try:
            llm = ChatDeepSeek(
                model=Config.DEEPSEEK_MODEL_NAME,      # 模型版本
                temperature=Config.LLM_TEMPERATURE,    # 生成温度
                api_key=Config.DEEPSEEK_API_KEY,       # API 密钥
            )
            logging.info(f"DeepSeek LLM (Direct) initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM (Direct): {e}")
            raise

    def get_llm(self):
        """
        获取底层 ChatDeepSeek 实例，供需要直接操作 LangChain LLM 的场景使用。

        返回：
            ChatDeepSeek: 底层 LLM 实例。
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
            # LangChain ChatDeepSeek 返回 AIMessage 对象，提取 .content 得到文本
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
