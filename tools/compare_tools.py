"""
模块：tools/compare_tools.py
职责：
    将 CompareSkills.compare_goods_parameters 封装为 LangChain Tool。

    Agent 传入逗号分隔的商品 ID 字符串，工具解析后调用技能
    查询数据库并返回格式化的多商品参数对比文本。

暴露接口：
    get_compare_tools(skill_router: SkillRouter) -> List[Tool]
"""

from typing import List
from langchain_core.tools import Tool

from agent_core.skill_router import SkillRouter


def get_compare_tools(skill_router: SkillRouter) -> List[Tool]:
    """
    构造商品对比工具列表。

    参数：
        skill_router: 已初始化的 SkillRouter 实例。

    返回：
        List[Tool]: 包含商品参数对比一个工具。
    """

    def compare_goods(query: str) -> str:
        # 支持中英文逗号分隔，过滤空字符串
        goods_ids = [g.strip() for g in query.replace("，", ",").split(",") if g.strip()]
        if len(goods_ids) < 2:
            return "请提供至少两个商品 ID 进行对比，用英文或中文逗号分隔。示例：G001,G002"
        result = skill_router.execute_skill("compare_goods_parameters", goods_ids=goods_ids)
        return result if isinstance(result, str) else str(result)

    return [
        Tool(
            name="compare_goods_parameters",
            func=compare_goods,
            description=(
                "对比多件商品的详细参数（价格、特点、优缺点、适用场景等）。"
                "适用：用户想比较几件具体商品，或推荐后需要横向对比时。"
                "输入：用英文或中文逗号分隔的商品 ID 列表，示例：G001,G002,G003"
            ),
        )
    ]
