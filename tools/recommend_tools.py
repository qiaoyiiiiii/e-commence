"""
模块：tools/recommend_tools.py
职责：
    将 RecommendSkills 的两个推荐方法封装为 LangChain Tool 对象，
    供 ReActAgentEngine 注册使用。

    封装层的核心职责：
      - 将 Agent 传入的自然语言字符串映射到技能函数所需的具体参数
      - 将技能返回的 List[Dict] 格式化为可读字符串（Agent 只能处理字符串观察值）
      - 为 Agent 提供清晰的工具描述，帮助其正确选择工具和构造输入

暴露接口：
    get_recommend_tools(user_id: str, skill_router: SkillRouter) -> List[Tool]
"""

from typing import List
from langchain_core.tools import Tool

from agent_core.skill_router import SkillRouter


def _fmt_goods(result) -> str:
    """将技能返回的商品列表统一格式化为可读字符串。"""
    if not result:
        return "未找到符合条件的商品。"
    if not isinstance(result, list):
        return str(result)
    lines = []
    for i, item in enumerate(result, 1):
        if isinstance(item, dict):
            name = item.get("name", "未知")
            goods_id = item.get("goods_id", "")
            price = item.get("price", "N/A")
            feature = item.get("feature", "")
            id_part = f"ID: {goods_id}，" if goods_id else ""
            suffix = f"  特点：{feature}" if feature else ""
            lines.append(f"{i}. {name}（{id_part}价格：{price} 元）{suffix}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def get_recommend_tools(user_id: str, skill_router: SkillRouter) -> List[Tool]:
    """
    构造推荐类工具列表。

    参数：
        user_id     : 当前用户 ID，捕获进闭包供个性化推荐使用。
        skill_router: 已初始化的 SkillRouter 实例，避免重复创建。

    返回：
        List[Tool]: 包含需求匹配推荐和个性化推荐两个工具。
    """

    def demand_matching(query: str) -> str:
        result = skill_router.execute_skill(
            "recommend_by_demand_matching", user_query=query, user_id=user_id
        )
        return _fmt_goods(result)

    def personalized(query: str, uid: str = user_id) -> str:
        # uid 由闭包捕获，Agent 传入的 query 内容不影响推荐结果
        result = skill_router.execute_skill(
            "recommend_by_personalized_preferences", user_id=uid
        )
        return _fmt_goods(result)

    return [
        Tool(
            name="recommend_by_demand_matching",
            func=demand_matching,
            description=(
                "根据用户自然语言需求搜索并推荐相关商品，直接返回结果。"
                "适用：用户需求明确、直接要求推荐时（如\"推荐一款通勤包\"）。"
                "不适用：用户需求模糊、对推荐质量有疑问、或需要分析理由时，那种情况请用 self_reflection_check。"
                "输入：用户的需求描述字符串。"
            ),
        ),
        Tool(
            name="recommend_by_personalized_preferences",
            func=personalized,
            description=(
                "根据用户已保存的个人偏好（颜色、风格、预算等）进行个性化推荐，"
                "无需用户再描述需求。"
                "适用：用户未指定明确需求，或要求根据个人喜好推荐时。"
                "输入：任意字符串（工具自动读取用户偏好，输入内容不影响结果）。"
            ),
        ),
    ]
