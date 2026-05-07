"""
模块：tools/tool_loader.py
职责：
    聚合所有工具模块，提供统一的工具加载入口 get_all_tools()。

    ReActAgentEngine 只需调用 get_all_tools(user_id) 即可获取完整的工具列表，
    无需了解各工具的具体实现细节。新增工具类别时只需：
      1. 在 tools/ 目录下新建对应模块
      2. 在本文件的 get_all_tools() 中调用并合并

架构说明：
    tools/ 目录是 skills/ 与 Agent 之间的适配层：
        skills/  →  业务逻辑（SQL 查询、RAG 检索、LLM 调用）
        tools/   →  Agent 接口（字符串 I/O、错误处理、格式化）
        react_agent.py → Agent 编排（不关心工具细节）

暴露接口：
    get_all_tools(user_id: str) -> List[Tool]
"""

import logging
from typing import List

from langchain_core.tools import Tool

from agent_core.skill_router import SkillRouter
from tools.recommend_tools import get_recommend_tools
from tools.filter_tools import get_filter_tools
from tools.compare_tools import get_compare_tools
from tools.check_tools import get_check_tools
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")


def get_all_tools(user_id: str) -> List[Tool]:
    """
    初始化 SkillRouter 并聚合所有可用工具，返回完整工具列表。

    所有工具模块共享同一个 SkillRouter 实例，避免 Skills（尤其是
    RecommendSkills 内的 HybridRetrieverManager）被重复初始化。

    参数：
        user_id (str): 当前用户 ID，传递给需要用户上下文的工具（如个性化推荐）。

    返回：
        List[Tool]: 所有已注册工具的列表，可直接传入 AgentExecutor。
    """
    # 单次初始化 SkillRouter，所有工具模块共享，防止重复加载模型
    skill_router = SkillRouter()

    tools: List[Tool] = []
    tools.extend(get_recommend_tools(user_id, skill_router))
    tools.extend(get_filter_tools(skill_router))
    tools.extend(get_compare_tools(skill_router))
    tools.extend(get_check_tools(skill_router))

    logging.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    return tools
