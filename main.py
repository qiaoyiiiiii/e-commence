"""
模块职责：
    电商购物 Agent 的主入口文件，负责：
    1. 验证数据库连接，确保系统依赖就绪。
    2. 初始化 MCPManager（内部懒加载 ReActAgentEngine、记忆管理等组件）。
    3. 启动命令行交互界面（CLI），将用户输入委托给 MCPManager.process_user_input()，
       由其完成记忆管理、偏好增强、ReAct 工具决策和最终回答的完整流程。

架构说明：
    main.py 只负责 CLI 交互和生命周期管理，不直接操作 LLM、检索器或工具。
    所有业务逻辑均下沉到以下层：
        MCPManager         → 对话流程协调（记忆 + 偏好 + Agent 调度）
        ReActAgentEngine   → ReAct 循环推理（Thought/Action/Observation）
        tools/             → Skills 的 Agent 接口适配层
        skills/            → 具体业务逻辑（推荐、过滤、对比、反思）

依赖：
    - config.Config              : 全局配置参数
    - database.Database          : MySQL 数据库单例（连接检查）
    - agent_core.mcp_manager.MCPManager : 对话协调中心

使用方式：
    直接运行：
        python main.py
        python main.py --user_id alice
"""

import argparse
import logging
from dotenv import load_dotenv

from config import Config
from database import Database
from agent_core.mcp_manager import MCPManager

# 加载 .env 文件中的环境变量（需在其他模块导入之前调用，确保配置生效）
load_dotenv()

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_agent_cli(user_id: str = "default_user"):
    """
    启动电商购物 Agent 的命令行交互界面（CLI 模式）。

    参数：
        user_id (str): 当前用户的唯一标识符，用于隔离不同用户的记忆数据。
                       默认值为 "default_user"，适用于单用户本地测试场景。
    """
    logger.info(f"Starting {Config.PROJECT_NAME} CLI for user: {user_id}...")

    # 步骤1：验证数据库连接。Database() 为单例，首次调用时自动建库建表。
    db = Database()
    if not db.connection or not db.connection.is_connected():
        logger.error("Database connection failed. Exiting.")
        return

    # 步骤2：初始化 MCPManager（加载用户长期记忆；ReActAgentEngine 在首次对话时懒加载）
    mcp_manager = MCPManager(user_id)

    print("\n" + "=" * 50)
    print(f"欢迎使用 {Config.PROJECT_NAME}！")
    print("输入 'exit' 或 'quit' 退出。")
    print("=" * 50 + "\n")

    # 步骤3：主交互循环
    while True:
        user_input = input("您有什么需求？ ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("感谢您的使用，再见！")
            break

        logger.info(f"User query: {user_input}")
        try:
            # MCPManager 内部完成：记忆写入 → 历史构建 → 偏好增强 → ReAct 推理 → 记忆压缩
            response = mcp_manager.process_user_input(user_input)

            print("\n" + "-" * 50)
            print("Agent 回复:")
            print(response)
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            print("抱歉，Agent 在处理您的请求时遇到了问题。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{Config.PROJECT_NAME} CLI")
    parser.add_argument(
        "--user_id",
        type=str,
        default="default_user",
        help="User ID for memory management",
    )
    args = parser.parse_args()
    run_agent_cli(args.user_id)
