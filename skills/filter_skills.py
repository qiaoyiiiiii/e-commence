"""
模块：skills/filter_skills.py
职责：
    提供基于多维条件的商品过滤能力（FilterSkills），以及对单件商品进行约束校验的能力。
    核心功能：
      - 根据价格区间、品类、品牌、JSON 数组字段（场景/人群/风格/标签）动态拼接 SQL 查询
      - 支持用户禁止标签（forbidden_tags）和禁止商品 ID（forbidden_goods_ids）的反向过滤
      - 对指定商品 ID 校验其价格是否满足给定约束

依赖：
    - database.Database   : 封装了 MySQL 连接与查询执行的数据库工具类
    - config.Config       : 项目全局配置，包含日志级别等参数

使用方式：
    from skills.filter_skills import FilterSkills
    fs = FilterSkills()
    results = fs.filter_goods_by_criteria(max_price=200, category="包包", scene=["通勤"])
    is_ok   = fs.validate_constraints("G001", {"max_price": 180, "min_price": 100})
"""

import logging
import json
from typing import List, Dict, Any, Optional

from database import Database
from config import Config

# 使用项目统一的日志级别和格式初始化日志记录器
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class FilterSkills:
    """
    商品过滤技能类。

    封装了通过多种维度（价格、品类、品牌、标签、场景等）筛选商品的能力，
    以及对单件商品进行约束条件校验的能力。
    内部使用动态 SQL 构建器，将筛选条件安全地转换为参数化查询，防止 SQL 注入。

    属性：
        db (Database): 数据库连接实例，用于执行所有 SQL 查询。
    """

    def __init__(self):
        """
        初始化 FilterSkills，创建数据库连接实例。
        """
        # 创建数据库连接，后续所有查询均通过此实例执行
        self.db = Database()

    def _build_sql_query(self, base_query: str, filters: Dict[str, Any]) -> (str, List[Any]):
        """
        根据过滤条件字典，在基础 SQL 语句后动态拼接 WHERE 子句。

        设计原则：
          - 所有条件均通过 %s 占位符传递，避免 SQL 注入
          - JSON 数组字段使用 MySQL 的 JSON_CONTAINS 函数进行包含匹配
          - 禁止条件使用 NOT JSON_CONTAINS 或 NOT IN 实现反向过滤
          - 多个条件之间用 AND 连接（即各条件取交集）

        参数：
            base_query (str): 不含 WHERE 子句的基础 SELECT 语句。
            filters (dict): 过滤条件字典，仅包含非 None 的字段，支持的键：
                - "min_price"         (float): 最低价格（含）
                - "max_price"         (float): 最高价格（含）
                - "category"          (str)  : 商品品类，精确匹配
                - "brand"             (str)  : 商品品牌，精确匹配
                - "scene"             (list) : 适用场景列表，需全部包含
                - "person"            (list) : 适用人群列表，需全部包含
                - "style"             (list) : 风格列表，需全部包含
                - "tags"              (list) : 标签列表，需全部包含
                - "forbidden_tags"    (list) : 禁止出现的标签列表
                - "forbidden_goods_ids" (list): 禁止返回的商品 ID 列表

        返回：
            tuple[str, list]: (完整 SQL 字符串, 对应参数列表)
                参数列表与 SQL 中 %s 占位符一一对应，需以 tuple 形式传入 execute_query。
        """
        conditions = []  # 存储所有 WHERE 子条件字符串
        params = []      # 存储与条件对应的参数值，顺序必须与占位符一致

        # --- 数值范围过滤 ---
        if "min_price" in filters:
            conditions.append("price >= %s")
            params.append(filters["min_price"])
        if "max_price" in filters:
            conditions.append("price <= %s")
            params.append(filters["max_price"])

        # --- 精确匹配字段过滤 ---
        if "category" in filters:
            conditions.append("category = %s")
            params.append(filters["category"])
        if "brand" in filters:
            conditions.append("brand = %s")
            params.append(filters["brand"])

        # --- JSON 数组字段的包含匹配 ---
        # MySQL JSON_CONTAINS(json_col, JSON_QUOTE(value)) 检查数组中是否包含某个字符串元素。
        # 对列表中的每个值各生成一条 AND 条件，要求商品同时满足所有指定值（交集语义）。
        for json_field in ["scene", "person", "style", "tags"]:
            if json_field in filters and filters[json_field]:
                for value in filters[json_field]:
                    conditions.append(f"JSON_CONTAINS({json_field}, JSON_QUOTE(%s))")
                    params.append(value)

        # --- 禁止标签的反向过滤 ---
        # 对每个禁止标签生成 NOT JSON_CONTAINS 条件，排除包含这些标签的商品
        if "forbidden_tags" in filters and filters["forbidden_tags"]:
            for tag in filters["forbidden_tags"]:
                conditions.append(f"NOT JSON_CONTAINS(tags, JSON_QUOTE(%s))")
                params.append(tag)

        # --- 禁止商品 ID 的反向过滤 ---
        # 使用 NOT IN 子句排除指定 goods_id 的商品，动态生成等量占位符
        if "forbidden_goods_ids" in filters and filters["forbidden_goods_ids"]:
            placeholders = ', '.join(['%s'] * len(filters["forbidden_goods_ids"]))
            conditions.append(f"goods_id NOT IN ({placeholders})")
            params.extend(filters["forbidden_goods_ids"])

        # --- 拼接完整查询语句 ---
        full_query = base_query
        if conditions:
            # 将所有子条件用 AND 连接追加在 WHERE 之后
            full_query += " WHERE " + " AND ".join(conditions)

        return full_query, params

    def filter_goods_by_criteria(self,
                                 min_price: Optional[float] = None,
                                 max_price: Optional[float] = None,
                                 category: Optional[str] = None,
                                 brand: Optional[str] = None,
                                 scene: Optional[List[str]] = None,
                                 person: Optional[List[str]] = None,
                                 style: Optional[List[str]] = None,
                                 tags: Optional[List[str]] = None,
                                 forbidden_tags: Optional[List[str]] = None,
                                 forbidden_goods_ids: Optional[List[str]] = None,
                                 limit: int = 10
                                ) -> List[Dict[str, Any]]:
        """
        根据多种筛选条件从数据库中检索商品列表。

        所有参数均为可选；未传入的参数不会对查询产生影响（不生成对应 WHERE 条件）。
        JSON 数组字段（scene/person/style/tags）在数据库中以 JSON 字符串存储，
        查询结果中会自动反序列化为 Python list。

        参数：
            min_price (float, 可选): 商品最低价格（含）。
            max_price (float, 可选): 商品最高价格（含）。
            category  (str,  可选): 商品品类，精确匹配。
            brand     (str,  可选): 商品品牌，精确匹配。
            scene     (list, 可选): 适用场景列表，商品需同时包含所有指定场景。
            person    (list, 可选): 适用人群列表，商品需同时包含所有指定人群。
            style     (list, 可选): 风格列表，商品需同时包含所有指定风格。
            tags      (list, 可选): 标签列表，商品需同时包含所有指定标签。
            forbidden_tags      (list, 可选): 禁止出现的标签，命中任一则排除该商品。
            forbidden_goods_ids (list, 可选): 明确排除的商品 ID 列表。
            limit     (int):  返回结果的最大条数，默认 10。

        返回：
            list[dict]: 满足条件的商品字典列表，每个字典包含以下字段：
                goods_id, name, category, price, brand, scene, person,
                style, tags, feature, advantage, disadvantage。
                scene/person/style/tags 均已反序列化为 Python list。
            若无匹配结果则返回空列表 []。
        """
        logging.info(f"Filtering goods with criteria: min_price={min_price}, max_price={max_price}, category={category}, ...")

        # 将所有参数打包为字典，便于统一传递给 _build_sql_query
        filters = {
            "min_price": min_price,
            "max_price": max_price,
            "category": category,
            "brand": brand,
            "scene": scene,
            "person": person,
            "style": style,
            "tags": tags,
            "forbidden_tags": forbidden_tags,
            "forbidden_goods_ids": forbidden_goods_ids,
        }

        # 基础 SELECT 语句，指定需要返回的字段
        base_query = "SELECT goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage FROM goods"

        # 仅将值不为 None 的条件传入构建器，避免生成多余的 WHERE 子句
        query, params = self._build_sql_query(base_query, {k: v for k, v in filters.items() if v is not None})

        # 追加 LIMIT 限制，防止返回过多数据影响性能
        query += f" LIMIT %s"
        params.append(limit)

        # 执行查询，fetch_type='all' 返回全部匹配行
        results = self.db.execute_query(query, tuple(params), fetch_type='all')
        if results:
            # 将数据库中以 JSON 字符串存储的数组字段反序列化为 Python list
            for good in results:
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key) and isinstance(good[key], str):
                        good[key] = json.loads(good[key])
            logging.info(f"Found {len(results)} goods after filtering.")
            return results

        logging.info("No goods found matching the criteria.")
        return []

    def validate_constraints(self, goods_id: str, constraints: Dict[str, Any]) -> bool:
        """
        校验指定商品是否满足给定的约束条件。

        目前支持的约束：
          - "max_price" (float): 商品价格不得超过该值
          - "min_price" (float): 商品价格不得低于该值

        参数：
            goods_id    (str):  待校验的商品唯一标识符。
            constraints (dict): 约束条件字典，支持 "max_price" 和 "min_price" 键。

        返回：
            bool: 商品满足所有约束时返回 True，否则返回 False。
                  若 goods_id 在数据库中不存在，也返回 False。

        注意：
            此方法仅从数据库查询 goods_id、name、price、category 四个字段，
            未来如需扩展品类/品牌等约束校验，可在此方法中追加逻辑。
        """
        logging.info(f"Validating constraints for goods_id {goods_id} with constraints: {constraints}")

        # 仅查询校验所需的最小字段集，减少数据传输量
        query = "SELECT goods_id, name, price, category FROM goods WHERE goods_id = %s"
        good = self.db.execute_query(query, (goods_id,), fetch_type='one')

        if not good:
            # 商品不存在，无法通过校验
            logging.warning(f"Goods_id {goods_id} not found for constraint validation.")
            return False

        # --- 价格上限校验 ---
        if "max_price" in constraints and good["price"] > constraints["max_price"]:
            return False  # 商品价格超出用户预算上限

        # --- 价格下限校验 ---
        if "min_price" in constraints and good["price"] < constraints["min_price"]:
            return False  # 商品价格低于用户要求的最低价格

        # 所有约束均满足
        return True


# ---------------------------------------------------------------------------
# 模块独立运行示例（仅用于开发调试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 运行前请确保 MySQL 已启动并已填充示例数据（参考 database.py）
    filter_skills = FilterSkills()

    print("\n--- Filtering for bags under 200 with '通勤' scene ---")
    filtered_bags = filter_skills.filter_goods_by_criteria(
        max_price=200.0,
        category="包包",
        scene=["通勤"]
    )
    for good in filtered_bags:
        print(f"- {good['name']} ({good['goods_id']}), Price: {good['price']}, Scene: {good['scene']}")

    print("\n--- Filtering for items tagged '性价比' and not '红色' (forbidden) ---")
    filtered_by_tags = filter_skills.filter_goods_by_criteria(
        tags=["性价比"],
        forbidden_tags=["红色"],  # 假设 '红色' 可能是某商品的标签
        limit=5
    )
    for good in filtered_by_tags:
        print(f"- {good['name']} ({good['goods_id']}), Tags: {good['tags']}")

    print("\n--- Validating constraints for a specific good (G001) ---")
    is_valid = filter_skills.validate_constraints("G001", {"max_price": 180.0, "min_price": 100.0})
    print(f"Is G001 valid under constraints (price between 100 and 180)? {is_valid}")

    is_valid_fail = filter_skills.validate_constraints("G001", {"max_price": 100.0})
    print(f"Is G001 valid under constraints (max price 100)? {is_valid_fail}")
