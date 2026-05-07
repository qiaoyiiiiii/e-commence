"""
模块：skills/compare_skills.py
职责：
    提供商品参数对比能力（CompareSkills）。
    核心功能：
      - 根据商品 ID 列表从数据库批量查询商品完整信息
      - 将多件商品的关键参数（类别、品牌、价格、特点、优缺点、适用场景等）格式化为
        易于阅读的多行文本，供 Agent 直接呈现给用户

依赖：
    - database.Database : 封装了 MySQL 连接与查询执行的数据库工具类
    - config.Config     : 项目全局配置，包含日志级别等参数

使用方式：
    from skills.compare_skills import CompareSkills
    cs = CompareSkills()
    summary = cs.compare_goods_parameters(["G001", "G002"])
    print(summary)
"""

import logging
from typing import List, Dict, Any, Optional
import json

from database import Database
from config import Config

# 使用项目统一的日志级别和格式初始化日志记录器
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class CompareSkills:
    """
    商品参数对比技能类。

    封装了从数据库批量获取商品详情，并将多件商品的核心参数
    格式化为结构化对比文本的能力，方便用户直观比较不同商品。

    属性：
        db (Database): 数据库连接实例，用于执行商品查询。
    """

    def __init__(self):
        """
        初始化 CompareSkills，创建数据库连接实例。
        """
        # 创建数据库连接，后续所有查询均通过此实例执行
        self.db = Database()

    def get_goods_details(self, goods_ids: List[str]) -> List[Dict[str, Any]]:
        """
        根据商品 ID 列表批量查询商品的完整详情。

        使用 IN 子句一次性查询多件商品，避免 N+1 查询问题。
        查询结果中，scene/person/style/tags 字段若以 JSON 字符串存储，
        将自动反序列化为 Python list；解析失败时保留原值并记录警告。

        参数：
            goods_ids (list[str]): 待查询的商品 ID 列表。
                                   若列表为空，直接返回空列表，不执行数据库查询。

        返回：
            list[dict]: 商品详情字典列表，每个字典包含 goods 表的所有字段（SELECT *）。
                        scene/person/style/tags 字段已尽量反序列化为 Python list。
                        若数据库中无匹配记录则返回空列表 []。
        """
        # 防御性检查：空列表无需查询数据库
        if not goods_ids:
            return []

        # 动态生成 IN 子句的占位符，数量与 goods_ids 长度一致
        placeholders = ', '.join(['%s'] * len(goods_ids))
        query = f"SELECT * FROM goods WHERE goods_id IN ({placeholders})"

        # 以 tuple 形式传入参数，执行批量查询
        results = self.db.execute_query(query, tuple(goods_ids), fetch_type='all')

        if results:
            # 对每件商品的 JSON 数组字段进行反序列化
            for good in results:
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key) and isinstance(good[key], str):
                        try:
                            good[key] = json.loads(good[key])
                        except json.JSONDecodeError:
                            # JSON 格式异常时记录警告，保留原始字符串值，不中断流程
                            logging.warning(f"Could not decode JSON for {key} in good {good.get('goods_id')}")
                            pass  # 保留原始字符串，由调用方决定如何处理
            logging.info(f"Retrieved details for {len(results)} goods.")
            return results

        logging.info(f"No goods found for IDs: {goods_ids}")
        return []

    def compare_goods_parameters(self, goods_ids: List[str]) -> str:
        """
        对多件商品的关键参数进行对比，返回格式化的对比文本。

        对比内容包括：类别、品牌、价格、特点、优点、缺点、适用场景、适用人群、风格、标签。
        每件商品独立成块，以分隔线区分，便于用户阅读和比较。

        参数：
            goods_ids (list[str]): 待对比的商品 ID 列表，建议传入 2~5 件。

        返回：
            str: 多行格式化的商品对比文本。
                 若所有 goods_ids 均无法在数据库中找到对应记录，
                 则返回提示字符串 "没有找到可供比较的商品。"。

        示例输出：
            以下是您选择的商品的对比：

            --- 经典款运动鞋 (C001) ---
              - 类别: 鞋子
              - 品牌: 运动品牌A
              - 价格: 399 元
              ...
        """
        logging.info(f"Comparing goods with IDs: {goods_ids}")

        # 批量获取所有待对比商品的详情
        goods_details = self.get_goods_details(goods_ids)

        # 若没有找到任何商品，返回友好提示
        if not goods_details:
            return "没有找到可供比较的商品。"

        # 拼接对比文本：每件商品占一个块，以 "---" 分隔线标识商品名称和 ID
        comparison_summary = "以下是您选择的商品的对比：\n\n"
        for good in goods_details:
            # 商品标题行：显示名称和 ID，未知时使用占位符
            comparison_summary += f"--- {good.get('name', '未知商品')} ({good.get('goods_id', '未知ID')}) ---\n"

            # 基础属性：类别、品牌、价格
            comparison_summary += f"  - 类别: {good.get('category', '-')}\n"
            comparison_summary += f"  - 品牌: {good.get('brand', '-')}\n"
            comparison_summary += f"  - 价格: {good.get('price', '-')} 元\n"

            # 商品描述：特点、优点、缺点
            comparison_summary += f"  - 特点: {good.get('feature', '-')}\n"
            comparison_summary += f"  - 优点: {good.get('advantage', '-')}\n"
            comparison_summary += f"  - 缺点: {good.get('disadvantage', '-')}\n"

            # JSON 数组字段：用逗号+空格连接列表元素后输出
            # good.get(key, []) 确保字段缺失时不会抛出异常
            comparison_summary += f"  - 适用场景: {', '.join(good.get('scene', []))}\n"
            comparison_summary += f"  - 适用人群: {', '.join(good.get('person', []))}\n"
            comparison_summary += f"  - 风格: {', '.join(good.get('style', []))}\n"
            comparison_summary += f"  - 标签: {', '.join(good.get('tags', []))}\n"

            # 商品块之间添加空行，提升可读性
            comparison_summary += "\n"

        return comparison_summary

    # 预留接口：基于 LLM 的优缺点智能总结（计划中）
    # def summarize_pros_cons(self, goods_ids: List[str]) -> str:
    #     """ 调用 LLM 对选定商品的优缺点进行智能归纳总结。 """
    #     pass


# ---------------------------------------------------------------------------
# 模块独立运行示例（仅用于开发调试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 运行前请确保 MySQL 已启动并已填充示例商品数据
    compare_skills = CompareSkills()

    # 若测试数据不存在，先插入两条示例商品
    db_instance = Database()
    sample_good1 = {
        "goods_id": "C001",
        "name": "经典款运动鞋",
        "category": "鞋子",
        "price": 399,
        "brand": "运动品牌A",
        "scene": ["运动", "休闲"],
        "person": ["男士", "女士"],
        "style": ["运动", "时尚"],
        "tags": ["透气", "舒适"],
        "feature": "轻量化设计",
        "advantage": "适合长时间穿着",
        "disadvantage": "不防水"
    }
    sample_good2 = {
        "goods_id": "C002",
        "name": "时尚休闲板鞋",
        "category": "鞋子",
        "price": 299,
        "brand": "时尚品牌B",
        "scene": ["休闲", "日常"],
        "person": ["青少年", "学生"],
        "style": ["休闲", "潮流"],
        "tags": ["百搭", "个性"],
        "feature": "独特外观设计",
        "advantage": "搭配多种服饰",
        "disadvantage": "支撑性一般"
    }

    insert_good_query = (
        "INSERT IGNORE INTO goods (goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    try:
        db_instance.execute_query(insert_good_query, (
            sample_good1["goods_id"], sample_good1["name"], sample_good1["category"], sample_good1["price"], sample_good1["brand"],
            json.dumps(sample_good1["scene"]), json.dumps(sample_good1["person"]), json.dumps(sample_good1["style"]),
            json.dumps(sample_good1["tags"]), sample_good1["feature"], sample_good1["advantage"], sample_good1["disadvantage"]),
        )
        db_instance.execute_query(insert_good_query, (
            sample_good2["goods_id"], sample_good2["name"], sample_good2["category"], sample_good2["price"], sample_good2["brand"],
            json.dumps(sample_good2["scene"]), json.dumps(sample_good2["person"]), json.dumps(sample_good2["style"]),
            json.dumps(sample_good2["tags"]), sample_good2["feature"], sample_good2["advantage"], sample_good2["disadvantage"]),
        )
        print("Sample comparison goods ensured to exist.")
    except Exception as e:
        print(f"Error inserting sample comparison goods: {e}")

    print("\n--- Comparing goods C001 and C002 ---")
    comparison_result = compare_skills.compare_goods_parameters(["C001", "C002"])
    print(comparison_result)

    print("\n--- Comparing non-existent goods ---")
    comparison_result_empty = compare_skills.compare_goods_parameters(["NON_EXISTENT_G001", "NON_EXISTENT_G002"])
    print(comparison_result_empty)
