"""
模块：tools/check_tools.py
职责：
    将 CheckSkills.self_reflection_check 封装为带检索能力的 LangChain Tool。

    工具接收用户需求字符串，内部先通过 RecommendSkills 的检索器召回候选商品，
    再将 (user_query, recommended_goods) 一并送入 self_reflection_check 进行
    LLM 反思评估，最终返回改进建议。

    这是一个"检索 + 反思"的组合工具，而非仅对已有推荐结果做后置验证。
    工具层在内部补齐 goods 参数，对 Agent 暴露单一字符串接口。

暴露接口：
    get_check_tools(skill_router: SkillRouter) -> List[Tool]
"""

from typing import List
from langchain_core.tools import Tool

from agent_core.skill_router import SkillRouter


def get_check_tools(skill_router: SkillRouter) -> List[Tool]:
    """
    构造自我反思工具列表。

    参数：
        skill_router: 已初始化的 SkillRouter 实例。

    返回：
        List[Tool]: 包含自我反思检查一个工具。
    """

    def self_reflection(query: str) -> str:
        # 通过 RecommendSkills 实例的检索器召回候选商品，补齐 skill 所需的 goods 参数
        goods = []
        try:
            rs_func = skill_router._skills.get("recommend_by_demand_matching")
            if rs_func and hasattr(rs_func, "__self__"):
                docs = rs_func.__self__.retriever.invoke(query)
                goods = [
                    {
                        "name": d.metadata.get("name", ""),
                        "price": d.metadata.get("price", ""),
                        "feature": d.metadata.get("feature", ""),
                    }
                    for d in docs
                ]
        except Exception:
            pass  # 召回失败时以空列表调用 skill，skill 内部会返回相应提示

        result = skill_router.execute_skill(
            "self_reflection_check", user_query=query, recommended_goods=goods
        )
        return result or "反思完成，推荐结果无明显问题。"

    return [
        Tool(
            name="self_reflection_check",
            func=self_reflection,
            description=(
                "检索候选商品后，由 LLM 深度评估是否真正契合用户需求，并给出有理有据的推荐理由或改进建议。"
                "适用：用户需求模糊复杂、对推荐结果有疑问、或明确要求认真分析时。"
                "不适用：用户需求明确且只需快速给出结果时，那种情况请用 recommend_by_demand_matching。"
                "输入：用户的原始需求描述字符串。"
            ),
        )
    ]
