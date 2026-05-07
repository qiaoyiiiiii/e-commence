"""
模块职责：
    电商购物 Agent 的主入口文件，负责：
    1. 初始化 RAG（检索增强生成）链路所需的各个组件（LLM、向量检索器、Prompt 模板）。
    2. 启动命令行交互界面（CLI），接收用户自然语言输入并返回商品推荐回答。
    3. 管理每轮对话的短期记忆（短期记忆写入 → 历史摘要构建 → 记忆压缩）。
    4. 将用户的长期偏好与禁忌商品信息拼入查询，使检索结果更贴合个人需求。

依赖：
    - langchain / langchain_core：构建 RAG 链路（检索链 + 文档合并链）
    - config.Config：全局配置参数
    - database.Database：MySQL 数据库单例（连接检查）
    - agent_core.react_agent.DeepSeekLLM：DeepSeek 大语言模型封装
    - rag_module.hybrid_retriever.HybridRetrieverManager：混合检索器（向量 + 关键词）
    - data.prompt_templates：RAG Prompt 模板与对话历史格式化工具
    - agent_core.memory_manager.MemoryManager：短期/长期记忆管理
    - agent_core.mcp_manager.MCPManager：多组件协调管理器（含记忆、工具等）

使用方式：
    直接运行：
        python main.py
        python main.py --user_id alice

    在代码中调用：
        from main import run_agent_cli
        run_agent_cli(user_id="alice")
"""

import argparse
import logging
import os
from dotenv import load_dotenv

# LangChain 核心工具：输出解析器（将 LLM 原始输出转为字符串）
from langchain_core.output_parsers import StrOutputParser
# LangChain 核心工具：直通 Runnable（用于链式组合时透传数据）
from langchain_core.runnables import RunnablePassthrough
# 检索链：将检索器与文档合并链组合为完整的 RAG 流水线
from langchain.chains import create_retrieval_chain
# 文档合并链：将多个检索文档拼入 Prompt 后交给 LLM 生成答案
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import Config
from database import Database
from agent_core.react_agent import DeepSeekLLM
from rag_module.hybrid_retriever import HybridRetrieverManager
from data.prompt_templates import RAG_PROMPT_TEMPLATE, format_chat_history
from agent_core.memory_manager import MemoryManager
from agent_core.mcp_manager import MCPManager

# 加载 .env 文件中的环境变量（需在其他模块导入之前调用，确保配置生效）
load_dotenv()

# 初始化模块级别日志记录器，格式包含时间戳、模块名和日志级别
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_docs(docs):
    """
    将检索到的 Document 列表格式化为可直接嵌入 Prompt 的字符串。

    格式为：
        [来源:<商品名称>]
        <文档正文内容>

        [来源:<商品名称>]
        <文档正文内容>
        ...

    参数：
        docs (list[Document]): LangChain Document 对象列表，
                               每个对象包含 page_content 和 metadata 属性。

    返回：
        str: 用双空行分隔的多文档字符串，直接拼入 Prompt 的上下文区域。

    说明：
        metadata 中优先取 'name' 字段作为来源标识（对应商品名称），
        若缺失则显示 '未知'。
    """
    return "\n\n".join([
        # 每个文档块前缀标注商品来源，便于 LLM 引用和用户溯源
        f"[来源:{d.metadata.get('name', '未知')}]\n{d.page_content}"
        for d in docs
    ])


def initialize_rag_chain():
    """
    初始化 RAG 链路所需的三个核心组件：LLM、检索器、Prompt 模板。

    初始化顺序：
        1. 验证数据库连接（确保商品数据可查）
        2. 初始化 DeepSeek LLM（用于生成最终回答）
        3. 初始化混合检索器（向量检索 + 关键词检索）

    返回：
        tuple: (llm, retriever, rag_prompt)
            - llm: 已配置的 LangChain LLM 对象
            - retriever: 混合检索器对象，实现 BaseRetriever 接口
            - rag_prompt: RAG 专用 ChatPromptTemplate 对象

        若任一组件初始化失败，返回 (None, None, None)，
        调用方应检查返回值并中止后续流程。

    异常：
        内部捕获 ValueError（LLM 初始化失败，通常为 API Key 缺失）；
        其他异常不捕获，允许向上传播以便定位问题。
    """
    logger.info("Initializing RAG chain components...")

    # 步骤1：验证数据库连接。
    # Database() 为单例，首次调用时自动建库建表；此处仅做连接状态检查。
    db = Database()
    if not db.connection or not db.connection.is_connected():
        logger.error("Database connection failed. Exiting.")
        return None, None, None

    # 步骤2：初始化 DeepSeek LLM。
    # get_llm() 返回符合 LangChain BaseLLM 接口的 ChatDeepSeek 实例。
    try:
        deepseek_llm = DeepSeekLLM()
        llm = deepseek_llm.get_llm()
    except ValueError as e:
        # 最常见原因：DEEPSEEK_API_KEY 未配置
        logger.error(f"Failed to initialize DeepSeek LLM: {e}")
        return None, None, None

    # 步骤3：初始化混合检索器。
    # HybridRetrieverManager 内部会加载 ChromaDB 向量库和 BM25 关键词索引，
    # 并根据 Config.RERANKING_ENABLED 决定是否附加重排序层。
    hybrid_retriever_manager = HybridRetrieverManager()
    retriever = hybrid_retriever_manager.get_retriever()

    logger.info("RAG chain components initialized successfully.")
    return llm, retriever, RAG_PROMPT_TEMPLATE


def run_agent_cli(user_id: str = "default_user"):
    """
    启动电商购物 Agent 的命令行交互界面（CLI 模式）。

    参数：
        user_id (str): 当前用户的唯一标识符，用于隔离不同用户的记忆数据。
                       默认值为 "default_user"，适用于单用户本地测试场景。

    行为：
        1. 创建 MCPManager 实例，加载该用户的长期记忆（偏好、禁忌商品）。
        2. 初始化 RAG 链路（LLM + 检索器 + Prompt）。
        3. 组合为完整的 LangChain 检索链（retrieval_chain）。
        4. 进入交互循环：
            a. 读取用户输入
            b. 将用户消息写入短期记忆
            c. 构建包含历史摘要和近期对话的上下文块
            d. 将用户长期偏好拼入查询（增强个性化检索）
            e. 调用 RAG 链生成回答
            f. 打印回答和参考商品来源
            g. 将 Agent 回答写入短期记忆，必要时触发记忆压缩
        5. 用户输入 'exit' 或 'quit' 时退出循环。

    返回：
        None（CLI 函数，无返回值）

    异常：
        每轮对话的异常被捕获并打印友好提示，不中断整个会话循环。
    """
    logger.info(f"Starting {Config.PROJECT_NAME} CLI for user: {user_id}...")

    # 初始化 MCPManager：内部创建 MemoryManager（加载用户长期记忆），
    # 并以懒加载方式准备其他工具组件。
    mcp_manager = MCPManager(user_id)

    # 初始化 RAG 链路的三个核心组件
    llm, retriever, rag_prompt = initialize_rag_chain()
    if llm is None or retriever is None or rag_prompt is None:
        # 任一组件失败则无法正常服务，直接退出
        logger.error("Agent initialization failed. Please check logs for details.")
        return

    # create_stuff_documents_chain：将检索文档"塞入"（stuff）Prompt，
    # 再调用 LLM 生成答案；适用于文档总量不超过上下文窗口的场景。
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    # create_retrieval_chain：将检索器与文档合并链串联为完整 RAG 流水线。
    # 调用 invoke({"input": ...}) 时会自动完成：检索 → 文档合并 → LLM 生成。
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # 打印欢迎界面
    print("\n" + "=" * 50)
    print(f"欢迎使用 {Config.PROJECT_NAME}！")
    print("输入 'exit' 或 'quit' 退出。")
    print("=" * 50 + "\n")

    # 主交互循环：持续接收用户输入直到退出指令
    while True:
        user_input = input("您有什么需求？ ")

        # 检测退出指令，不区分大小写
        if user_input.lower() in ['exit', 'quit']:
            print("感谢您的使用，再见！")
            break

        logger.info(f"User query: {user_input}")
        try:
            # ---------------------------------------------------------------
            # 步骤1：将用户消息写入短期记忆（包含本轮消息的完整历史）
            # ---------------------------------------------------------------
            mcp_manager.memory_manager.add_message_to_short_term_memory("user", user_input)

            # ---------------------------------------------------------------
            # 步骤2：构建对话历史上下文块
            # 策略：优先使用 LLM 生成的摘要（处理长对话），
            #       再拼接最近几轮的原始对话（保持细节准确性）。
            # [:-1] 切片：排除刚刚写入的当前用户消息，避免在历史中重复出现。
            # ---------------------------------------------------------------
            history = mcp_manager.memory_manager.get_short_term_memory()[:-1]
            history_summary = mcp_manager.memory_manager.get_history_summary()
            recent_str = format_chat_history(history)

            # 根据摘要和近期对话是否存在，组合不同格式的历史上下文
            if history_summary and recent_str:
                # 摘要 + 近期对话均存在：两者合并提供最完整的上下文
                chat_history_block = f"[历史摘要] {history_summary}\n[最近对话]\n{recent_str}\n\n"
            elif history_summary:
                # 仅有摘要（近期消息已全部被压缩）：直接使用摘要
                chat_history_block = f"[历史摘要] {history_summary}\n\n"
            elif recent_str:
                # 仅有近期对话（对话轮数尚未触发压缩阈值）：直接使用原始历史
                chat_history_block = f"对话历史：\n{recent_str}\n\n"
            else:
                # 首轮对话，无任何历史记录
                chat_history_block = ""

            # ---------------------------------------------------------------
            # 步骤3：将用户长期偏好拼入查询，增强个性化检索效果。
            # memory_hint 包含用户偏好关键词和禁忌商品提示，
            # 拼入查询后向量检索会更倾向于匹配符合偏好的商品文档。
            # ---------------------------------------------------------------
            memory_hint = mcp_manager.get_memory_context_hint()
            # 若有记忆提示则将其附加到原始查询末尾，否则直接使用原始查询
            query_input = f"{user_input}。[{memory_hint}]" if memory_hint else user_input

            # ---------------------------------------------------------------
            # 步骤4：调用 RAG 链，获取回答和参考文档
            # 输入字典说明：
            #   - "input"：经过偏好增强的查询字符串（用于检索和 LLM 生成）
            #   - "chat_history"：格式化后的历史上下文（注入 Prompt 的对话历史槽位）
            # 输出字典说明：
            #   - "answer"：LLM 生成的最终回答文本
            #   - "context"：检索到的 Document 对象列表（含商品 metadata）
            # ---------------------------------------------------------------
            response = rag_chain.invoke({"input": query_input, "chat_history": chat_history_block})
            answer = response["answer"]
            context_docs = response["context"]  # 检索返回的参考商品文档列表

            # ---------------------------------------------------------------
            # 步骤5：将 Agent 回答写入短期记忆，并按需触发压缩
            # compress_short_term_memory 会检查消息总数是否超过
            # Config.MEMORY_COMPRESSION_THRESHOLD，超过则调用 LLM 生成摘要
            # 并丢弃较早的原始消息，只保留最近 MEMORY_COMPRESSION_KEEP_RECENT 条。
            # ---------------------------------------------------------------
            mcp_manager.memory_manager.add_message_to_short_term_memory("agent", answer)
            mcp_manager.memory_manager.compress_short_term_memory(llm)

            # 打印 Agent 回答和参考来源
            print("\n" + "-" * 50)
            print("Agent 回复:")
            print(answer)
            print("\n参考资料:")
            for doc in context_docs:
                # 优先展示商品名称，若缺失则回退到 goods_id，最终兜底为"未知"
                source_name = doc.metadata.get('name', doc.metadata.get('goods_id', '未知'))
                # 仅截取正文前 50 个字符预览，避免终端输出过长
                print(f"- {source_name}: {doc.page_content[:50]}...")
            print("-" * 50 + "\n")

        except Exception as e:
            # 捕获单轮对话中的所有异常，保证会话循环不中断
            logger.error(f"Error during RAG chain invocation: {e}")
            print("抱歉，Agent 在处理您的请求时遇到了问题。")


if __name__ == "__main__":
    # 解析命令行参数，支持通过 --user_id 指定用户 ID
    parser = argparse.ArgumentParser(description=f"{Config.PROJECT_NAME} CLI")
    parser.add_argument(
        "--user_id",
        type=str,
        default="default_user",
        help="User ID for memory management"  # 用于隔离不同用户的记忆数据
    )
    args = parser.parse_args()

    # 使用解析到的 user_id 启动 CLI 主循环
    run_agent_cli(args.user_id)
