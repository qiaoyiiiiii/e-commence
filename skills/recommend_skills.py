import logging
import json
from typing import List, Dict, Any, Optional

from database import Database
from config import Config
from agent_core.memory_manager import MemoryManager
from rag_module.hybrid_retriever import HybridRetrieverManager

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class RecommendSkills:
    def __init__(self):
        self.db = Database()
        self.hybrid_retriever_manager = HybridRetrieverManager()
        self.retriever = self.hybrid_retriever_manager.get_retriever()

    def recommend_by_demand_matching(self, 
                                     user_query: str,
                                     user_id: str = "default_user",
                                     limit: int = Config.RECOMMENDATION_COUNT
                                    ) -> List[Dict[str, Any]]:
        """ Recommends goods based on user's query using RAG (demand matching). """
        logging.info(f"Recommending by demand matching for user '{user_id}' with query: '{user_query}'")
        
        # Retrieve relevant documents (goods) based on the user query
        # The retriever already incorporates embeddings and reranking
        retrieved_docs = self.retriever.invoke(user_query)
        
        recommended_goods = []
        for doc in retrieved_docs:
            # Assuming metadata contains all original good details
            good_data = doc.metadata
            # Convert JSON string fields back to Python objects if they were not already
            for key in ['scene', 'person', 'style', 'tags']:
                if good_data.get(key) and isinstance(good_data[key], str):
                    try:
                        good_data[key] = json.loads(good_data[key])
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON for {key} in good {good_data.get('goods_id')}")
                        pass # Keep as string if decoding fails
            recommended_goods.append(good_data)
            if len(recommended_goods) >= limit:
                break

        logging.info(f"Found {len(recommended_goods)} recommendations for query '{user_query}'.")
        return recommended_goods

    def recommend_by_personalized_preferences(self,
                                              user_id: str,
                                              limit: int = Config.RECOMMENDATION_COUNT
                                             ) -> List[Dict[str, Any]]:
        """ Recommends goods based on user's long-term preferences. """
        logging.info(f"Recommending by personalized preferences for user '{user_id}'")
        memory_manager = MemoryManager(user_id)
        preferences = memory_manager.get_user_preferences()
        forbidden_items = memory_manager.get_forbidden_items()

        # This is a simplified approach. A real implementation would convert preferences
        # into a query that the RAG system can understand or use specific filtering.
        # For now, let's construct a basic query from preferences.
        preference_query_parts = []
        if preferences.get('favorite_color'):
            preference_query_parts.append(f"{preferences['favorite_color']}颜色的商品")
        if preferences.get('budget'):
            preference_query_parts.append(f"{preferences['budget']}价位的商品")
        # Add more preference-based query parts

        # Combine with a generic search if no specific preferences are found
        if not preference_query_parts:
            preference_query = "推荐一些热门商品"
        else:
            preference_query = "、".join(preference_query_parts) + "。请推荐相关商品。"
        
        logging.info(f"Generated preference query: {preference_query}")
        
        # Use the existing RAG retriever with the generated preference query
        retrieved_docs = self.retriever.invoke(preference_query)

        recommended_goods = []
        for doc in retrieved_docs:
            good_data = doc.metadata
            # Apply forbidden items filter (simplified)
            is_forbidden = False
            if forbidden_items:
                for forbidden_item in forbidden_items:
                    # Very basic check, needs more sophistication
                    if forbidden_item.lower() in good_data.get('name', '').lower() or \
                       forbidden_item.lower() in good_data.get('tags', '').lower():
                        is_forbidden = True
                        break
            if not is_forbidden:
                # Convert JSON string fields back to Python objects if they were not already
                for key in ['scene', 'person', 'style', 'tags']:
                    if good_data.get(key) and isinstance(good_data[key], str):
                        try:
                            good_data[key] = json.loads(good_data[key])
                        except json.JSONDecodeError:
                            logging.warning(f"Could not decode JSON for {key} in good {good_data.get('goods_id')}")
                            pass # Keep as string if decoding fails
                recommended_goods.append(good_data)
                if len(recommended_goods) >= limit:
                    break

        logging.info(f"Found {len(recommended_goods)} personalized recommendations for user '{user_id}'.")
        return recommended_goods

    # Future: Implement pairing recommendations
    # def recommend_pairing(self, goods_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    #     """ Recommends complementary items for a given product. """
    #     pass

# Example usage
if __name__ == "__main__":
    # Ensure MySQL is running, populated with sample data, and .env is configured
    recommend_skills = RecommendSkills()

    # Test demand matching
    print("\n--- Demand Matching Recommendation ---")
    demand_recs = recommend_skills.recommend_by_demand_matching("给我推荐一款适合上班族用的简约风格包包", user_id="test_user_001", limit=2)
    for rec in demand_recs:
        print(f"- {rec.get('name')}, Price: {rec.get('price')}, Category: {rec.get('category')}")

    # Test personalized recommendations
    # First, set some preferences for test_user_001
    memory_manager_test = MemoryManager("test_user_001")
    memory_manager_test.update_long_term_memory('preferences', {"favorite_color": "蓝色", "budget": "中等"})
    memory_manager_test.update_long_term_memory('forbidden_items', ["红色", "运动鞋"])

    print("\n--- Personalized Recommendation ---")
    personalized_recs = recommend_skills.recommend_by_personalized_preferences("test_user_001", limit=3)
    for rec in personalized_recs:
        print(f"- {rec.get('name')}, Price: {rec.get('price')}, Category: {rec.get('category')}")
