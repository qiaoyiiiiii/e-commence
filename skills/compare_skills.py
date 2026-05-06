import logging
from typing import List, Dict, Any, Optional
import json

from database import Database
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class CompareSkills:
    def __init__(self):
        self.db = Database()

    def get_goods_details(self, goods_ids: List[str]) -> List[Dict[str, Any]]:
        """ Retrieves detailed information for a list of goods_ids. """
        if not goods_ids:
            return []
        
        placeholders = ', '.join(['%s'] * len(goods_ids))
        query = f"SELECT * FROM goods WHERE goods_id IN ({placeholders})"
        results = self.db.execute_query(query, tuple(goods_ids), fetch_type='all')

        if results:
            for good in results:
                # Convert JSON string fields back to Python objects
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key) and isinstance(good[key], str):
                        try:
                            good[key] = json.loads(good[key])
                        except json.JSONDecodeError:
                            logging.warning(f"Could not decode JSON for {key} in good {good.get('goods_id')}")
                            pass
            logging.info(f"Retrieved details for {len(results)} goods.")
            return results
        logging.info(f"No goods found for IDs: {goods_ids}")
        return []

    def compare_goods_parameters(self, goods_ids: List[str]) -> str:
        """ Compares key parameters of multiple goods and returns a summary. """
        logging.info(f"Comparing goods with IDs: {goods_ids}")
        goods_details = self.get_goods_details(goods_ids)

        if not goods_details:
            return "没有找到可供比较的商品。"

        comparison_summary = "以下是您选择的商品的对比：\n\n"
        for good in goods_details:
            comparison_summary += f"--- {good.get('name', '未知商品')} ({good.get('goods_id', '未知ID')}) ---\n"
            comparison_summary += f"  - 类别: {good.get('category', '-')}\n"
            comparison_summary += f"  - 品牌: {good.get('brand', '-')}\n"
            comparison_summary += f"  - 价格: {good.get('price', '-')} 元\n"
            comparison_summary += f"  - 特点: {good.get('feature', '-')}\n"
            comparison_summary += f"  - 优点: {good.get('advantage', '-')}\n"
            comparison_summary += f"  - 缺点: {good.get('disadvantage', '-')}\n"
            comparison_summary += f"  - 适用场景: {', '.join(good.get('scene', []))}\n"
            comparison_summary += f"  - 适用人群: {', '.join(good.get('person', []))}\n"
            comparison_summary += f"  - 风格: {', '.join(good.get('style', []))}\n"
            comparison_summary += f"  - 标签: {', '.join(good.get('tags', []))}\n"
            comparison_summary += "\n"

        return comparison_summary

    # Future: Implement more advanced pros/cons summarization using LLM
    # def summarize_pros_cons(self, goods_ids: List[str]) -> str:
    #     """ Summarizes pros and cons of selected goods, potentially using LLM. """
    #     pass

# Example usage
if __name__ == "__main__":
    # Ensure MySQL is running and populated with sample goods data
    compare_skills = CompareSkills()

    # Insert some dummy data if not already present (for testing)
    db_instance = Database()
    sample_good1 = {
        "goods_id": "C001",
        "name": "经典款运动鞋",
        "category": "鞋子",
        "price": 399,
        "brand": "运动品牌A",
        "scene": ["运动","休闲"],
        "person": ["男士","女士"],
        "style": ["运动","时尚"],
        "tags": ["透气","舒适"],
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
        "scene": ["休闲","日常"],
        "person": ["青少年","学生"],
        "style": ["休闲","潮流"],
        "tags": ["百搭","个性"],
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
