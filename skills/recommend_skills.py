"""
模块：skills/recommend_skills.py
职责：
    提供商品推荐能力（RecommendSkills），支持两种推荐策略：
      1. 基于需求匹配的推荐（recommend_by_demand_matching）：
         接收用户自然语言查询，通过 RAG 混合检索器召回语义相关商品。
      2. 基于个性化偏好的推荐（recommend_by_personalized_preferences）：
         读取用户长期记忆中的偏好与禁止商品，自动构造检索查询并过滤禁止项后推荐。

依赖：
    - database.Database                    : 数据库连接工具（当前类中保留，供未来扩展）
    - config.Config                        : 项目全局配置（RECOMMENDATION_COUNT、LOG_LEVEL 等）
    - agent_core.memory_manager.MemoryManager : 用户长期记忆读写接口
    - rag_module.hybrid_retriever.HybridRetrieverManager : 融合向量检索与关键词检索的混合检索器

使用方式：
    from skills.recommend_skills import RecommendSkills
    rs = RecommendSkills()
    goods = rs.recommend_by_demand_matching("适合上班族的简约包包", user_id="u001")
    goods = rs.recommend_by_personalized_preferences("u001")
"""

import logging
import json
from typing import List, Dict, Any, Optional

from database import Database
from config import Config
from agent_core.memory_manager import MemoryManager
from rag_module.hybrid_retriever import HybridRetrieverManager

# 使用项目统一的日志级别和格式初始化日志记录器
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class RecommendSkills:
    """
    商品推荐技能类。

    封装了两种推荐路径：
      - 需求匹配推荐：将用户查询直接送入 RAG 混合检索器，利用向量语义相似度召回最相关商品。
      - 个性化偏好推荐：从用户长期记忆中读取偏好，自动生成检索查询，
        并在结果中过滤用户明确禁止的商品（按名称和标签匹配）。

    属性：
        db (Database): 数据库连接实例（保留，便于未来直接查询扩展）。
        hybrid_retriever_manager (HybridRetrieverManager): 混合检索器管理器。
        retriever: 由管理器返回的具体检索器实例，实现 .invoke(query) 接口。
    """

    def __init__(self):
        """
        初始化 RecommendSkills：
          - 创建数据库连接
          - 初始化混合检索器管理器并获取检索器实例
        注意：HybridRetrieverManager 初始化时会加载向量库，首次调用可能较慢。
        """
        self.db = Database()
        # 混合检索器管理器：融合向量检索与 BM25 关键词检索
        self.hybrid_retriever_manager = HybridRetrieverManager()
        # 获取具体检索器，后续推荐方法均通过 self.retriever.invoke() 召回文档
        self.retriever = self.hybrid_retriever_manager.get_retriever()

    def _parse_json_fields(self, good_data: Dict[str, Any]) -> None:
        """
        将商品字典中以 JSON 字符串存储的数组字段原地反序列化为 Python list。

        数据库中 scene、person、style、tags 字段以 JSON 字符串形式存储（如 '["通勤","休闲"]'），
        检索器返回的 metadata 中这些字段可能仍为字符串，需反序列化后才能正常使用。

        参数：
            good_data (dict): 单件商品的元数据字典，函数直接修改此字典（原地操作）。

        返回：
            None（原地修改 good_data，无返回值）。

        异常处理：
            若某字段的 JSON 解析失败（格式非法），记录警告日志并将该字段置为空列表，
            不会抛出异常以保证推荐流程继续执行。
        """
        for key in ['scene', 'person', 'style', 'tags']:
            if good_data.get(key) and isinstance(good_data[key], str):
                try:
                    good_data[key] = json.loads(good_data[key])
                except json.JSONDecodeError:
                    # 记录警告但不中断流程，将问题字段设为空列表
                    logging.warning(f"Could not decode JSON for {key} in good {good_data.get('goods_id')}")
                    good_data[key] = []

    def recommend_by_demand_matching(self,
                                     user_query: str,
                                     user_id: str = "default_user",
                                     limit: int = Config.RECOMMENDATION_COUNT
                                    ) -> List[Dict[str, Any]]:
        """
        基于用户自然语言需求，通过 RAG 混合检索器进行商品推荐。

        工作流程：
          1. 将 user_query 传入混合检索器（向量 + BM25），召回语义相关的商品文档。
          2. 从文档的 metadata 中提取商品信息。
          3. 对 JSON 字符串字段进行反序列化。
          4. 按召回顺序返回前 limit 件商品。

        参数：
            user_query (str): 用户输入的自然语言需求描述，例如"适合上班的简约包包"。
            user_id    (str): 当前用户 ID，当前仅用于日志记录，默认 "default_user"。
            limit      (int): 返回推荐商品的最大数量，默认取 Config.RECOMMENDATION_COUNT。

        返回：
            list[dict]: 推荐商品列表，每个元素为包含商品详情的字典（来自检索文档的 metadata）。
                        scene/person/style/tags 已反序列化为 Python list。
                        若检索结果为空则返回空列表 []。
        """
        logging.info(f"Recommending by demand matching for user '{user_id}' with query: '{user_query}'")

        # 调用混合检索器召回与 user_query 语义最相关的商品文档列表
        # 检索器内部已整合向量相似度计算和重排序（reranking）
        retrieved_docs = self.retriever.invoke(user_query)

        recommended_goods = []
        for doc in retrieved_docs:
            # 检索文档的 metadata 中存储了完整的商品字段信息
            good_data = doc.metadata
            # 将 JSON 字符串字段反序列化为 Python list（原地修改）
            self._parse_json_fields(good_data)
            recommended_goods.append(good_data)
            # 达到数量上限后停止遍历
            if len(recommended_goods) >= limit:
                break

        logging.info(f"Found {len(recommended_goods)} recommendations for query '{user_query}'.")
        return recommended_goods

# 根据长期数据获得统计偏好和厌恶预测货物
    def recommend_by_personalized_preferences(self,
                                              user_id: str,
                                              limit: int = Config.RECOMMENDATION_COUNT
                                             ) -> List[Dict[str, Any]]:
        """
        基于用户长期记忆中的个性化偏好进行商品推荐，并自动过滤用户禁止的商品。

        工作流程：
          1. 从 MemoryManager 读取用户偏好（favorite_color、budget 等）和禁止商品列表。
          2. 根据偏好字段拼接自然语言检索查询（无偏好时使用通用热门推荐查询）。
          3. 将生成的查询送入混合检索器召回候选商品。
          4. 对每件候选商品，检查其名称和标签是否包含禁止关键词，命中则跳过。
          5. 通过禁止过滤的商品经 JSON 反序列化后加入推荐列表，直至达到 limit。

        参数：
            user_id (str): 目标用户 ID，用于从 MemoryManager 读取其长期偏好数据。
            limit   (int): 返回推荐商品的最大数量，默认取 Config.RECOMMENDATION_COUNT。

        返回：
            list[dict]: 个性化推荐商品列表，已过滤用户禁止项，JSON 字段已反序列化。
                        若无符合条件的商品则返回空列表 []。

        注意：
            - 偏好查询构造为简化实现，仅使用 favorite_color 和 budget 两个字段。
              如需更精细的个性化，可扩展此处的查询构造逻辑。
            - 禁止过滤采用字符串包含匹配（大小写不敏感），非精确 ID 匹配，
              因此可能存在误判，未来可结合 goods_id 白名单/黑名单优化。
        """
        logging.info(f"Recommending by personalized preferences for user '{user_id}'")

        # 创建该用户的内存管理器实例，读取长期偏好与禁止商品
        memory_manager = MemoryManager(user_id)
        preferences = memory_manager.get_user_preferences()    # 用户偏好字典
        forbidden_items = memory_manager.get_forbidden_items()  # 用户禁止商品关键词列表

        # --- 根据用户偏好构造检索查询字符串 ---
        # 简化实现：目前仅支持 favorite_color 和 budget 两个偏好维度
        preference_query_parts = []
        if preferences.get('favorite_color'):
            # 将颜色偏好转为自然语言查询片段
            preference_query_parts.append(f"{preferences['favorite_color']}颜色的商品")
        if preferences.get('budget'):
            # 将预算偏好转为自然语言查询片段
            preference_query_parts.append(f"{preferences['budget']}价位的商品")

        # 若用户无任何有效偏好记录，回退到通用热门商品查询
        if not preference_query_parts:
            preference_query = "推荐一些热门商品"
        else:
            # 用顿号拼接多个偏好片段，形成完整的检索查询
            preference_query = "、".join(preference_query_parts) + "。请推荐相关商品。"

        logging.info(f"Generated preference query: {preference_query}")

        # 调用混合检索器，以偏好查询字符串召回候选商品
        retrieved_docs = self.retriever.invoke(preference_query)

        recommended_goods = []
        for doc in retrieved_docs:
            good_data = doc.metadata

            # --- 禁止商品过滤 ---
            # 检查候选商品的名称或标签中是否包含用户禁止的关键词
            is_forbidden = False
            if forbidden_items:
                for forbidden_item in forbidden_items:
                    # 兼容 tags 字段为 list 或字符串两种格式
                    tags_value = good_data.get('tags', [])
                    if isinstance(tags_value, list):
                        # list 格式：将各元素拼接为空格分隔的字符串后统一转小写
                        tags_str = ' '.join(str(t) for t in tags_value).lower()
                    else:
                        # 字符串格式（JSON 字符串或普通字符串）：直接转小写
                        tags_str = str(tags_value).lower()

                    # 在商品名称和标签中查找禁止关键词（大小写不敏感）
                    if forbidden_item.lower() in good_data.get('name', '').lower() or \
                       forbidden_item.lower() in tags_str:
                        is_forbidden = True
                        break  # 命中任一禁止词即标记为禁止，无需继续检查

            if not is_forbidden:
                # 未命中禁止列表，进行 JSON 字段反序列化后加入推荐结果
                self._parse_json_fields(good_data)
                recommended_goods.append(good_data)
                if len(recommended_goods) >= limit:
                    break  # 已达推荐数量上限，停止遍历

        logging.info(f"Found {len(recommended_goods)} personalized recommendations for user '{user_id}'.")
        return recommended_goods


# ---------------------------------------------------------------------------
# 模块独立运行示例（仅用于开发调试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 运行前请确保 MySQL 已启动、向量库已初始化，且 .env 中已配置相关环境变量
    recommend_skills = RecommendSkills()

    # 测试需求匹配推荐
    print("\n--- Demand Matching Recommendation ---")
    demand_recs = recommend_skills.recommend_by_demand_matching("给我推荐一款适合上班族用的简约风格包包", user_id="test_user_001", limit=2)
    for rec in demand_recs:
        print(f"- {rec.get('name')}, Price: {rec.get('price')}, Category: {rec.get('category')}")

    # 测试个性化推荐（先为测试用户写入偏好数据）
    memory_manager_test = MemoryManager("test_user_001")
    memory_manager_test.update_long_term_memory('preferences', {"favorite_color": "蓝色", "budget": "中等"})
    memory_manager_test.update_long_term_memory('forbidden_items', ["红色", "运动鞋"])

    print("\n--- Personalized Recommendation ---")
    personalized_recs = recommend_skills.recommend_by_personalized_preferences("test_user_001", limit=3)
    for rec in personalized_recs:
        print(f"- {rec.get('name')}, Price: {rec.get('price')}, Category: {rec.get('category')}")
