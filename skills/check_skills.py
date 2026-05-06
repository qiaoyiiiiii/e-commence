import logging
from typing import List, Dict, Any

from config import Config
from agent_core.react_agent import DeepSeekLLM

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class CheckSkills:
    def __init__(self):
        self.llm_service = DeepSeekLLM() # Assuming DeepSeekLLM for self-reflection
        self.llm = self.llm_service.get_llm()

    def self_reflection_check(self, user_query: str, recommended_goods: List[Dict[str, Any]]) -> str:
        """ Performs a self-reflection check on the recommendations against the user query. """
        logging.info(f"Performing self-reflection for query: '{user_query}' and {len(recommended_goods)} goods.")

        if not recommended_goods:
            return "反思：没有推荐商品，可能需要重新评估用户需求或数据源。"

        # Construct a prompt for the LLM to perform self-reflection
        reflection_prompt = f"用户需求是：{user_query}\n"
        reflection_prompt += "推荐的商品有：\n"
        for good in recommended_goods:
            reflection_prompt += f"- 名称: {good.get('name', '')}, 价格: {good.get('price', '')}, 特点: {good.get('feature', '')}\n"
        reflection_prompt += "请评估这些推荐是否符合用户需求，是否有不合理之处，并给出你的反思和建议。"

        try:
            reflection_response = self.llm_service.invoke(reflection_prompt)
            logging.info("Self-reflection completed.")
            return reflection_response
        except Exception as e:
            logging.error(f"Error during self-reflection LLM invocation: {e}")
            return "反思：执行自省检查时发生错误。"

    # Future: Implement more advanced hallucination suppression and compliance checks
    # def hallucination_suppression(self, generated_text: str, context: List[str]) -> bool:
    #     """ Checks if the generated text contains information not supported by the context. """
    #     pass
    #
    # def compliance_check(self, recommendation: Dict[str, Any], rules: Dict[str, Any]) -> bool:
    #     """ Checks if a recommendation complies with predefined rules (e.g., legal, ethical). """
    #     pass

# Example usage
if __name__ == "__main__":
    # Ensure DEEPSEEK_API_KEY is set in .env
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
