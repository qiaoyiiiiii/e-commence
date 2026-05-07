"""
模块：data_processor.py
职责：
    负责从 MySQL 数据库加载商品数据和标签库数据，并将原始商品数据转换为
    LangChain Document 对象，以便后续写入向量数据库（ChromaDB）。

主要功能：
    1. filter_complex_metadata：对 metadata 字典做类型兼容性过滤，
       将 ChromaDB 不支持的类型（Decimal、list、dict、datetime 等）
       转换为合法的原生类型（float、str）。
    2. DataProcessor 类：封装数据库查询与文档构造逻辑。

依赖：
    - database.Database：项目封装的 MySQL 连接工具
    - langchain_core.documents.Document：LangChain 文档对象
    - config.Config：全局配置（日志级别等）

使用方式：
    processor = DataProcessor()
    documents = processor.load_and_process_goods()   # 获取可直接入库的文档列表
    tags      = processor.load_tag_library_from_mysql()  # 获取标签库
"""

import json
import os
import datetime
from decimal import Decimal
from typing import List, Dict, Any
from database import Database
from langchain_core.documents import Document
from config import Config
import logging

# 初始化日志，使用 Config 中定义的日志级别和格式
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 metadata 字典中的值过滤/转换为 ChromaDB 兼容的基本类型。

    ChromaDB 的 metadata 字段仅支持 str、int、float、bool 及 None；
    其他类型（Decimal、list、dict、datetime 等）需要提前转换，
    否则写入向量库时会抛出序列化错误。

    参数：
        metadata (Dict[str, Any])：原始 metadata 字典，值类型可能包含
            Python 任意类型。

    返回：
        Dict[str, Any]：所有值均为 ChromaDB 兼容类型的新字典。
            - None / bool / int / float / str  → 原样保留
            - Decimal                           → float
            - list / dict                       → json.dumps 序列化为字符串
            - datetime.datetime / datetime.date → ISO 8601 字符串
            - 其他类型                           → str(value)

    异常：
        无显式异常抛出；json.dumps 序列化失败时会由标准库抛出 TypeError。
    """
    filtered = {}
    for key, value in metadata.items():
        # None 以及已经是基本类型的值，直接保留
        if value is None or isinstance(value, (bool, int, float, str)):
            filtered[key] = value
        elif isinstance(value, Decimal):
            # MySQL 中 DECIMAL 字段读出后为 Python Decimal，需转为 float
            filtered[key] = float(value)
        elif isinstance(value, (list, dict)):
            # 列表/字典序列化为 JSON 字符串，保留中文字符（ensure_ascii=False）
            filtered[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            # 日期/时间类型转为 ISO 8601 标准字符串，例如 "2024-01-01T00:00:00"
            filtered[key] = value.isoformat()
        else:
            # 兜底：将其他未知类型强制转为字符串
            filtered[key] = str(value)
    return filtered


class DataProcessor:
    """
    数据处理器。

    负责：
        1. 从 MySQL 的 goods 表查询商品数据，并解析其中的 JSON 字段。
        2. 将商品字典列表转换为 LangChain Document 对象列表，
           同时对 metadata 做类型兼容处理。
        3. 从 MySQL 的 tag_library 表查询标签数据。

    属性：
        db (Database)：数据库连接实例（单例），用于执行 SQL 查询。
    """

    def __init__(self):
        """
        初始化 DataProcessor，创建数据库连接实例。
        """
        # Database() 为项目全局单例，负责管理 MySQL 连接池
        self.db = Database()

    def load_goods_data_from_mysql(self) -> List[Dict[str, Any]]:
        """
        从 MySQL goods 表中加载全量商品数据。

        查询 goods 表的所有记录，并对存储为 JSON 字符串的字段
        （scene、person、style、tags）执行 json.loads 反序列化，
        使其在后续处理中以 Python list 形式存在。

        返回：
            List[Dict[str, Any]]：商品字典列表，每个字典对应一条商品记录。
                JSON 字段已反序列化为 list；若表为空则返回空列表。

        副作用：
            - 成功时以 INFO 级别记录加载数量。
            - 无数据时以 WARNING 级别记录警告。
        """
        query = "SELECT * FROM goods"
        goods_data = self.db.execute_query(query, fetch_type='all')
        if goods_data:
            logging.info(f"Loaded {len(goods_data)} goods from MySQL.")
            # 将数据库中以 JSON 字符串存储的多值字段还原为 Python list
            for good in goods_data:
                for key in ['scene', 'person', 'style', 'tags']:
                    if good.get(key):
                        # 例如 '["夏季", "旅行"]' → ["夏季", "旅行"]
                        good[key] = json.loads(good[key])
            return goods_data
        logging.warning("No goods data found in MySQL.")
        return []

    def goods_to_langchain_documents(self, goods_data: List[Dict[str, Any]]) -> List[Document]:
        """
        将商品字典列表转换为 LangChain Document 对象列表。

        每条商品记录被转换为一个 Document：
            - page_content：由各字段拼接而成的自然语言文本，用于向量嵌入和语义检索。
            - metadata：商品原始字典的副本，经过 filter_complex_metadata 处理后
              保证与 ChromaDB 兼容。

        参数：
            goods_data (List[Dict[str, Any]])：由 load_goods_data_from_mysql 返回的
                商品字典列表，JSON 字段已反序列化。

        返回：
            List[Document]：可直接写入向量数据库的 LangChain Document 列表。

        副作用：
            以 INFO 级别记录转换数量。
        """
        documents = []
        for good in goods_data:
            # ── 构建 page_content ──────────────────────────────────────────
            # 固定字段：名称、类别、品牌、价格（必定存在，缺失时用空字符串兜底）
            content = f"名称: {good.get('name', '')}\n"
            content += f"类别: {good.get('category', '')}\n"
            content += f"品牌: {good.get('brand', '')}\n"
            content += f"价格: {good.get('price', '')}\n"

            # 可选文本字段：仅在数据库中有值时才追加，避免空行干扰向量质量
            if good.get('feature'):
                content += f"特点: {good.get('feature', '')}\n"
            if good.get('advantage'):
                content += f"优点: {good.get('advantage', '')}\n"
            if good.get('disadvantage'):
                content += f"缺点: {good.get('disadvantage', '')}\n"

            # 可选列表字段：list 元素用逗号拼接为自然语言短句
            if good.get('scene'):
                content += f"适用场景: {', '.join(good['scene'])}\n"
            if good.get('person'):
                content += f"适用人群: {', '.join(good['person'])}\n"
            if good.get('style'):
                content += f"风格: {', '.join(good['style'])}\n"
            if good.get('tags'):
                content += f"标签: {', '.join(good['tags'])}\n"

            # ── 构建 Document 并过滤 metadata ────────────────────────────
            # 使用 good.copy() 避免后续 filter_complex_metadata 修改原始数据
            doc = Document(page_content=content, metadata=good.copy())
            # 将 metadata 中不兼容 ChromaDB 的类型（Decimal、list 等）转换为合法类型
            doc.metadata = filter_complex_metadata(doc.metadata)
            documents.append(doc)

        logging.info(f"Converted {len(documents)} goods into LangChain Documents.")
        return documents

    def load_and_process_goods(self) -> List[Document]:
        """
        一站式加载并处理商品数据的入口方法。

        内部依次调用：
            1. load_goods_data_from_mysql：从数据库读取原始商品记录。
            2. goods_to_langchain_documents：将记录转换为 Document 列表。

        返回：
            List[Document]：可直接写入向量数据库的文档列表。
                若数据库中无商品数据，返回空列表。

        说明：
            当前每条商品对应一个 Document（整体入库）。若商品描述文本
            过长，可在此处引入 TextSplitter 进行分块处理。
        """
        goods_data = self.load_goods_data_from_mysql()
        if not goods_data:
            # 数据库无数据时提前返回，避免空转
            return []

        # 当前策略：每条商品作为一个完整 Document，不做分块
        # 若未来商品描述很长，可在此引入 RecursiveCharacterTextSplitter
        documents = self.goods_to_langchain_documents(goods_data)
        return documents

    def load_tag_library_from_mysql(self) -> List[Dict[str, Any]]:
        """
        从 MySQL tag_library 表加载标签库数据。

        标签库用于辅助意图识别、用户偏好提取等上层业务逻辑，
        本方法仅负责原始数据的读取，不做额外处理。

        返回：
            List[Dict[str, Any]]：标签字典列表，每个字典对应一条标签记录。
                若表为空则返回空列表。

        副作用：
            - 成功时以 INFO 级别记录加载数量。
            - 无数据时以 WARNING 级别记录警告。
        """
        query = "SELECT * FROM tag_library"
        tag_data = self.db.execute_query(query, fetch_type='all')
        if tag_data:
            logging.info(f"Loaded {len(tag_data)} tags from MySQL.")
            return tag_data
        logging.warning("No tag library data found in MySQL.")
        return []


# ── 模块独立运行时的简单演示 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 运行前提：MySQL 已启动，数据库/表已创建，.env 配置正确
    processor = DataProcessor()

    # 测试商品数据加载与转换
    goods_docs = processor.load_and_process_goods()
    print(f"\n--- Processed Goods Documents ({len(goods_docs)}) ---")
    for i, doc in enumerate(goods_docs[:2]):  # 仅打印前 2 条以便快速验证
        print(f"Document {i+1}:\nPage Content: {doc.page_content}\nMetadata: {doc.metadata}\n")

    # 测试标签库加载
    tag_library = processor.load_tag_library_from_mysql()
    print(f"\n--- Tag Library ({len(tag_library)}) ---")
    for tag in tag_library[:2]:  # 仅打印前 2 条
        print(f"Tag: {tag}\n")

    # 注：Database 为单例，程序退出时连接会自动释放；
    # 如需提前释放，可调用 self.db.close()（若该方法已实现）。
