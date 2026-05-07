"""
模块：tools/check_tools.py
职责：
    将 CheckSkills.self_reflection_check 封装为 LangChain Tool。

    Agent 传入用户的原始需求字符串，工具内部通过 RecommendSkills 的
    检索器重新召回候选商品，再将查询与商品列表一并送入 self_reflection_check
    进行 LLM 反思评估，返回评估文本。

    这样设计的原因：
        self_reflection_check 需要 (user_query, recommended_goods) 两个参数，
        但 ReAct Agent 的 Action Input 只能传字符串。工具层在内部补齐 goods 参数，
        对 Agent 暴露单一字符串接口，对 Skill 提供结构化参数。

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
                "对当前推荐结果进行自我反思，评估是否真正符合用户需求，给出改进建议。"
                "适用：完成推荐后需要验证质量，或用户对推荐结果有疑问时。"
                "输入：用户的原始需求描述字符串。"
            ),
        )
    ]
