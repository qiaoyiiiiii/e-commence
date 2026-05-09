"""
模块：skills/check_skills.py
职责：
    提供基于大语言模型（LLM）的推荐结果自省/自检能力（CheckSkills）。
    核心功能：
      - self_reflection_check：将用户原始需求与已推荐商品列表组合为反思提示词，
        调用 DeepSeek LLM 评估推荐结果是否合理，并返回反思建议文本。
    未来规划（已预留注释接口）：
      - hallucination_suppression：检测生成文本中是否包含上下文不支持的内容
      - compliance_check：校验推荐是否符合预定义的合规规则

依赖：
    - config.Config                     : 项目全局配置，包含日志级别等参数
    - agent_core.llm_client.DeepSeekLLM: 封装了 DeepSeek 大语言模型调用的服务类

使用方式：
    from skills.check_skills import CheckSkills
    cs = CheckSkills()
    reflection = cs.self_reflection_check("推荐夏天穿的连衣裙", goods_list)
    print(reflection)
"""

import logging
from typing import List, Dict, Any

from config import Config
from agent_core.llm_client import DeepSeekLLM

# 使用项目统一的日志级别和格式初始化日志记录器
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class CheckSkills:
    """
    推荐结果自省检查技能类。

    利用 DeepSeek 大语言模型对推荐结果进行自省评估：
    将用户需求和推荐商品列表构造为提示词，要求 LLM 分析推荐是否合理、
    是否存在不符合用户需求的商品，并给出改进建议。

    属性：
        _llm_service (DeepSeekLLM | None): DeepSeek LLM 服务实例，懒加载，初始为 None。
            仅在 self_reflection_check 被实际调用时才初始化，避免 SkillRouter
            注册阶段无谓地创建 LLM 客户端对象。
    """

    def __init__(self):
        """
        初始化 CheckSkills。LLM 不在此处创建，由 llm_service 属性懒加载。
        """
        self._llm_service: DeepSeekLLM | None = None

    @property
    def llm_service(self) -> DeepSeekLLM:
        """懒加载 DeepSeekLLM，首次调用 self_reflection_check 时才实际初始化。"""
        if self._llm_service is None:
            self._llm_service = DeepSeekLLM()
        return self._llm_service

    def self_reflection_check(self, user_query: str, recommended_goods: List[Dict[str, Any]]) -> str:
        """
        对推荐结果执行 LLM 自省检查，评估推荐是否符合用户需求。

        工作流程：
          1. 检查推荐商品列表是否为空，若为空则直接返回固定提示，不调用 LLM。
          2. 将用户需求和每件商品的名称、价格、特点拼接为反思提示词。
          3. 调用 DeepSeek LLM 对提示词进行推理，获取反思文本。
          4. 捕获 LLM 调用异常，记录错误日志并返回错误提示字符串。

        参数：
            user_query       (str):       用户的原始需求描述文本。
            recommended_goods (list[dict]): 已推荐给用户的商品字典列表，
                                            每个字典应至少包含 'name'、'price'、'feature' 字段。

        返回：
            str: LLM 生成的反思评估文本。
                 - 若 recommended_goods 为空：返回固定提示 "反思：没有推荐商品，可能需要重新评估用户需求或数据源。"
                 - 若 LLM 调用成功：返回 LLM 的反思建议内容。
                 - 若 LLM 调用失败：返回错误提示 "反思：执行自省检查时发生错误。"

        异常：
            本方法内部捕获所有 LLM 调用异常（Exception），不向上抛出，
            确保自省检查失败不会中断整体推荐流程。
        """
        logging.info(f"Performing self-reflection for query: '{user_query}' and {len(recommended_goods)} goods.")

        # 边界情况处理：无推荐商品时无需调用 LLM，直接返回提示
        if not recommended_goods:
            return "反思：没有推荐商品，可能需要重新评估用户需求或数据源。"

        # --- 构造反思提示词 ---
        # 提示词格式：先陈述用户需求，再列举每件推荐商品的关键信息，最后要求 LLM 评估
        reflection_prompt = f"用户需求是：{user_query}\n"
        reflection_prompt += "推荐的商品有：\n"
        for good in recommended_goods:
            # 对每件商品提取名称、价格、特点三个关键字段（缺失时使用空字符串）
            reflection_prompt += f"- 名称: {good.get('name', '')}, 价格: {good.get('price', '')}, 特点: {good.get('feature', '')}\n"
        reflection_prompt += "请评估这些推荐是否符合用户需求，是否有不合理之处，并给出你的反思和建议。"

        # --- 调用 LLM 执行反思推理 ---
        try:
            # 通过 llm_service.invoke() 发送提示词并获取 LLM 的文本响应
            reflection_response = self.llm_service.invoke(reflection_prompt)
            logging.info("Self-reflection completed.")
            return reflection_response
        except Exception as e:
            # 捕获所有异常（网络错误、API 限流、超时等），记录错误并返回友好提示
            logging.error(f"Error during self-reflection LLM invocation: {e}")
            return "反思：执行自省检查时发生错误。"

    # 预留接口：幻觉抑制检查（计划中）
    # def hallucination_suppression(self, generated_text: str, context: List[str]) -> bool:
    #     """ 检查生成文本中是否包含上下文不支持的信息（幻觉内容）。 """
    #     pass
    #
    # 预留接口：合规性检查（计划中）
    # def compliance_check(self, recommendation: Dict[str, Any], rules: Dict[str, Any]) -> bool:
    #     """ 检查推荐结果是否符合预定义的规则（如法律、伦理约束）。 """
    #     pass


# ---------------------------------------------------------------------------
# 模块独立运行示例（仅用于开发调试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 运行前请确保 .env 中已设置 DEEPSEEK_API_KEY
    check_skills = CheckSkills()

    test_query = "给我推荐一个适合夏天穿的连衣裙，颜色要鲜艳一点"
    test_goods = [
        {"name": "碎花雪纺连衣裙", "price": 189, "feature": "轻薄透气，碎花图案", "goods_id": "G001"},
        {"name": "纯棉T恤", "price": 59, "feature": "舒适吸汗", "goods_id": "G002"}
    ]

    print("\n--- Performing Self-Reflection Check ---")
    reflection = check_skills.self_reflection_check(test_query, test_goods)
    print(reflection)

    print("\n--- Performing Self-Reflection Check with no recommendations ---")
    reflection_no_goods = check_skills.self_reflection_check(test_query, [])
    print(reflection_no_goods)
