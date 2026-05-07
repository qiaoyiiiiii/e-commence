"""
模块：tools/filter_tools.py
职责：
    将 FilterSkills 的两个方法封装为 LangChain Tool 对象。

    filter_goods_by_criteria：Agent 传入 JSON 字符串（含筛选条件），
        工具解析后调用技能执行 SQL 查询并返回格式化结果。
    validate_constraints：Agent 传入包含 goods_id 和约束条件的 JSON 字符串，
        工具解析后校验该商品是否满足约束。

暴露接口：
    get_filter_tools(skill_router: SkillRouter) -> List[Tool]
"""

import json
from typing import List
from langchain_core.tools import Tool

from agent_core.skill_router import SkillRouter


def _fmt_goods(result) -> str:
    if not result:
        return "未找到符合条件的商品。"
    if not isinstance(result, list):
        return str(result)
    lines = []
    for i, item in enumerate(result, 1):
        if isinstance(item, dict):
            name = item.get("name", "未知")
            price = item.get("price", "N/A")
            category = item.get("category", "")
            lines.append(f"{i}. {name}（{category}，价格：{price} 元）")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def get_filter_tools(skill_router: SkillRouter) -> List[Tool]:
    """
    构造过滤与约束校验工具列表。

    参数：
        skill_router: 已初始化的 SkillRouter 实例。

    返回：
        List[Tool]: 包含条件过滤和约束校验两个工具。
    """

    def filter_goods(query: str) -> str:
        try:
            params = json.loads(query)
        except (json.JSONDecodeError, TypeError):
            return (
                "输入格式错误，请提供 JSON 格式的筛选条件。"
                "示例：{\"max_price\": 200, \"category\": \"包包\", \"scene\": [\"通勤\"]}"
            )
        result = skill_router.execute_skill("filter_goods_by_criteria", **params)
        return _fmt_goods(result)

    def validate(query: str) -> str:
        try:
            data = json.loads(query)
            ok = skill_router.execute_skill(
                "validate_constraints",
                goods_id=data["goods_id"],
                constraints=data.get("constraints", {}),
            )
            return "该商品满足约束条件。" if ok else "该商品不满足约束条件。"
        except (json.JSONDecodeError, KeyError):
            return (
                "输入格式错误，示例："
                "{\"goods_id\": \"G001\", \"constraints\": {\"max_price\": 300}}"
            )

    return [
        Tool(
            name="filter_goods_by_criteria",
            func=filter_goods,
            description=(
                "按条件精确筛选商品（直接查数据库）。"
                "适用：用户明确指定价格范围、品类、场景、风格等具体限制时。"
                "输入：JSON 格式的筛选条件，支持字段："
                "max_price(最高价), min_price(最低价), category(品类), brand(品牌), "
                "scene(场景列表), style(风格列表), tags(标签列表)。"
                "示例：{\"max_price\": 300, \"category\": \"包包\", \"scene\": [\"通勤\"]}"
            ),
        ),
        Tool(
            name="validate_constraints",
            func=validate,
            description=(
                "校验某商品是否满足价格约束。"
                "适用：推荐后需确认商品价格是否在用户预算范围内时。"
                "输入：JSON 格式，示例：{\"goods_id\": \"G001\", \"constraints\": {\"max_price\": 300}}"
            ),
        ),
    ]
