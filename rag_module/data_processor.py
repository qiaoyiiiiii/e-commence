import json
import os
from typing import List, Dict, Any
from database import Database
from langchain_core.documents import Document
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self):
        self.db = Database()

    def load_goods_data_from_mysql(self) -> List[Dict[str, Any]]:
        """ Loads all goods data from the MySQL database. """
        query = "SELECT * FROM goods"
        goods_data = self.db.execute_query(query, fetch_type='all')
        if goods_data:
            logging.info(f"Loaded {len(goods_data)} goods from MySQL.")
            # Convert JSON strings back to Python objects
            for good in goods_data:
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key):
                        good[key] = json.loads(good[key])
            return goods_data
        logging.warning("No goods data found in MySQL.")
        return []

    def goods_to_langchain_documents(self, goods_data: List[Dict[str, Any]]) -> List[Document]:
        """ Converts goods data into LangChain Document objects. """
        documents = []
        for good in goods_data:
            # Construct page_content from relevant fields for embedding
            content = f"名称: {good.get('name', '')}\n"
            content += f"类别: {good.get('category', '')}\n"
            content += f"品牌: {good.get('brand', '')}\n"
            content += f"价格: {good.get('price', '')}\n"
            if good.get('feature'):
                content += f"特点: {good.get('feature', '')}\n"
            if good.get('advantage'):
                content += f"优点: {good.get('advantage', '')}\n"
            if good.get('disadvantage'):
                content += f"缺点: {good.get('disadvantage', '')}\n"
            if good.get('scene'):
                content += f"适用场景: {', '.join(good['scene'])}\n"
            if good.get('person'):
                content += f"适用人群: {', '.join(good['person'])}\n"
            if good.get('style'):
                content += f"风格: {', '.join(good['style'])}\n"
            if good.get('tags'):
                content += f"标签: {', '.join(good['tags'])}\n"

            # Create document with original metadata, then filter complex types
            doc = Document(page_content=content, metadata=good.copy())
            doc.metadata = filter_complex_metadata(doc.metadata)
            documents.append(doc)

        logging.info(f"Converted {len(documents)} goods into LangChain Documents.")
        return documents


    def load_and_process_goods(self) -> List[Document]:
        """ Loads goods data from MySQL and converts it into LangChain Documents. """
        goods_data = self.load_goods_data_from_mysql()
        if not goods_data:
            return []

        # Placeholder for data chunking/splitting if needed for very long product descriptions
        # For now, each product is treated as one document
        documents = self.goods_to_langchain_documents(goods_data)
        return documents

    def load_tag_library_from_mysql(self) -> List[Dict[str, Any]]:
        """ Loads tag library from MySQL. """
        query = "SELECT * FROM tag_library"
        tag_data = self.db.execute_query(query, fetch_type='all')
        if tag_data:
            logging.info(f"Loaded {len(tag_data)} tags from MySQL.")
            return tag_data
        logging.warning("No tag library data found in MySQL.")
        return []

# Example usage
if __name__ == "__main__":
    # This part requires a running MySQL instance and the database/tables to be set up
    # and potentially some sample data inserted (e.g., by running database.py's example block)
    processor = DataProcessor()
    goods_docs = processor.load_and_process_goods()
    print(f"\n--- Processed Goods Documents ({len(goods_docs)}) ---")
    for i, doc in enumerate(goods_docs[:2]): # Print first 2 docs
        print(f"Document {i+1}:\nPage Content: {doc.page_content}\nMetadata: {doc.metadata}\n")

    tag_library = processor.load_tag_library_from_mysql()
    print(f"\n--- Tag Library ({len(tag_library)}) ---")
    for tag in tag_library[:2]: # Print first 2 tags
        print(f"Tag: {tag}\n")

    # Don't forget to close the database connection, though the singleton handles it implicitly
    # when the program exits or can be explicitly closed if needed.
