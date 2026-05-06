import logging
import json
from typing import List, Dict, Any, Optional

from database import Database
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class FilterSkills:
    def __init__(self):
        self.db = Database()

    def _build_sql_query(self, base_query: str, filters: Dict[str, Any]) -> (str, List[Any]):
        conditions = []
        params = []

        if "min_price" in filters:
            conditions.append("price >= %s")
            params.append(filters["min_price"])
        if "max_price" in filters:
            conditions.append("price <= %s")
            params.append(filters["max_price"])
        if "category" in filters:
            conditions.append("category = %s")
            params.append(filters["category"])
        if "brand" in filters:
            conditions.append("brand = %s")
            params.append(filters["brand"])

        # For JSON array fields, use JSON_CONTAINS
        for json_field in ["scene", "person", "style", "tags"]:
            if json_field in filters and filters[json_field]:
                for value in filters[json_field]:
                    conditions.append(f"JSON_CONTAINS({json_field}, JSON_QUOTE(%s))")
                    params.append(value)

        # Forbidden items (example: can be based on tags or specific goods_id)
        if "forbidden_tags" in filters and filters["forbidden_tags"]:
            for tag in filters["forbidden_tags"]:
                conditions.append(f"NOT JSON_CONTAINS(tags, JSON_QUOTE(%s))")
                params.append(tag)
        if "forbidden_goods_ids" in filters and filters["forbidden_goods_ids"]:
            placeholders = ', '.join(['%s'] * len(filters["forbidden_goods_ids"]))
            conditions.append(f"goods_id NOT IN ({placeholders})")
            params.extend(filters["forbidden_goods_ids"])

        full_query = base_query
        if conditions:
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
        """ Filters goods based on various criteria. """
        logging.info(f"Filtering goods with criteria: min_price={min_price}, max_price={max_price}, category={category}, ...")

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

        base_query = "SELECT goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage FROM goods"
        query, params = self._build_sql_query(base_query, {k: v for k, v in filters.items() if v is not None})
        
        query += f" LIMIT %s"
        params.append(limit)

        results = self.db.execute_query(query, tuple(params), fetch_type='all')
        if results:
            # Convert JSON string fields back to Python objects for consistency
            for good in results:
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key) and isinstance(good[key], str):
                        good[key] = json.loads(good[key])
            logging.info(f"Found {len(results)} goods after filtering.")
            return results
        logging.info("No goods found matching the criteria.")
        return []

    def validate_constraints(self, goods_id: str, constraints: Dict[str, Any]) -> bool:
        """ Validates if a specific good meets given constraints. """
        logging.info(f"Validating constraints for goods_id {goods_id} with constraints: {constraints}")
        query = "SELECT goods_id, name, price, category FROM goods WHERE goods_id = %s"
        good = self.db.execute_query(query, (goods_id,), fetch_type='one')

        if not good:
            logging.warning(f"Goods_id {goods_id} not found for constraint validation.")
            return False

        if "max_price" in constraints and good["price"] > constraints["max_price"]:
            return False
        if "min_price" in constraints and good["price"] < constraints["min_price"]:
            return False
        # Add more constraint checks here (e.g., category, brand, etc.)

        return True

# Example usage
if __name__ == "__main__":
    # Ensure MySQL is running and populated with some sample data (e.g., from database.py example)
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
        forbidden_tags=["红色"], # Assuming '红色' could be a tag or a property
        limit=5
    )
    for good in filtered_by_tags:
        print(f"- {good['name']} ({good['goods_id']}), Tags: {good['tags']}")

    print("\n--- Validating constraints for a specific good (G001) ---")
    is_valid = filter_skills.validate_constraints("G001", {"max_price": 180.0, "min_price": 100.0})
    print(f"Is G001 valid under constraints (price between 100 and 180)? {is_valid}")

    is_valid_fail = filter_skills.validate_constraints("G001", {"max_price": 100.0})
    print(f"Is G001 valid under constraints (max price 100)? {is_valid_fail}")
