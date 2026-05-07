"""
模块职责：
    封装与 MySQL 数据库的所有交互，包括：
    - 数据库与数据表的自动初始化（首次运行时自动建库建表）
    - 单例模式的连接管理（全局共享同一个连接对象）
    - 统一的查询执行接口（支持查询、插入、更新、删除）
    - 断线自动重连机制

依赖：
    - mysql-connector-python：MySQL 官方 Python 驱动
    - config.Config：提供数据库连接参数

使用方式：
    from database import Database

    db = Database()  # 获取单例，首次调用时自动建库建表
    result = db.execute_query("SELECT * FROM goods", fetch_type='all')
    db.close()

数据表说明：
    - goods：商品信息表，存储 SKU 级别的商品属性
    - user_memory：用户记忆表，持久化用户的偏好、禁忌商品和对话历史
"""

import mysql.connector
from mysql.connector import Error
from config import Config
import logging

# 使用 Config 中统一配置的日志级别初始化日志，
# 格式为：时间戳 - 级别 - 消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class Database:
    """
    MySQL 数据库访问类，采用单例模式确保全局只维护一个数据库连接。

    单例模式实现：
        通过重写 __new__ 方法，将唯一实例存储在类变量 _instance 中。
        任何地方调用 Database() 都会返回同一个对象，避免重复建立连接。

    自动初始化：
        首次实例化时依次执行：
        1. 建库（若数据库不存在）
        2. 建立目标数据库连接
        3. 建表（若表不存在）
    """

    # 类级别的单例实例缓存，初始为 None
    _instance = None

    def __new__(cls):
        """
        单例工厂方法，控制实例的创建。

        返回：
            Database: 全局唯一的 Database 实例。

        行为：
            - 若 _instance 为 None（首次调用），创建新实例并触发数据库连接。
            - 若 _instance 已存在，直接返回已有实例，不重复初始化。
        """
        if cls._instance is None:
            # 首次调用：通过父类 __new__ 创建实际对象
            cls._instance = super(Database, cls).__new__(cls)
            # 初始化连接属性为 None，防止后续 connect() 中判断出错
            cls._instance.connection = None
            # 立即尝试建立数据库连接（包含建库、建表流程）
            cls._instance.connect()
        return cls._instance

    def connect(self):
        """
        建立 MySQL 数据库连接。

        行为：
            1. 检查当前连接是否有效；若已有活跃连接则跳过。
            2. 调用 _create_database_if_not_exists() 确保目标数据库存在。
            3. 使用 Config 中的参数建立到目标数据库的连接。
            4. 连接成功后调用 _create_tables_if_not_exists() 确保表结构存在。
            5. 若连接失败，将 self.connection 置为 None 并记录错误日志。

        异常：
            捕获 mysql.connector.Error，不向上抛出，但会记录错误日志。
        """
        if self.connection is None or not self.connection.is_connected():
            try:
                # 必须先确保数据库存在，否则直接指定 database= 参数会抛出
                # "Unknown database" 错误（MySQL 不会自动建库）
                self._create_database_if_not_exists()

                # 建立到指定数据库的正式连接
                self.connection = mysql.connector.connect(
                    host=Config.MYSQL_HOST,
                    database=Config.MYSQL_DB,
                    user=Config.MYSQL_USER,
                    password=Config.MYSQL_PASSWORD,
                    port=Config.MYSQL_PORT
                )

                if self.connection.is_connected():
                    logging.info(f"Successfully connected to MySQL database: {Config.MYSQL_DB}")
                    # 连接成功后立即确保所有业务表都已创建
                    self._create_tables_if_not_exists()
                else:
                    logging.error("Failed to connect to MySQL database.")
            except Error as e:
                logging.error(f"Error connecting to MySQL database: {e}")
                # 连接失败时将连接对象置空，避免后续调用使用无效连接
                self.connection = None

    def _create_database_if_not_exists(self):
        """
        检查目标数据库是否存在，若不存在则创建。

        实现原理：
            先不指定 database 参数建立一个临时连接（连接到 MySQL 默认系统库），
            再执行 CREATE DATABASE IF NOT EXISTS 语句，最后关闭临时连接。
            这样可以避免直接连接不存在的数据库时抛出异常。

        字符集：
            使用 utf8mb4 + utf8mb4_unicode_ci，完整支持中文及 emoji。

        异常：
            遇到 mysql.connector.Error 时记录日志并向上重新抛出，
            以便 connect() 感知失败并停止后续步骤。
        """
        try:
            # 建立不指定具体数据库的临时连接，用于执行建库 DDL
            temp_conn = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                port=Config.MYSQL_PORT
            )
            cursor = temp_conn.cursor()

            # IF NOT EXISTS 保证重复执行不会报错
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{Config.MYSQL_DB}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

            # 及时释放临时游标和连接，避免占用 MySQL 连接数
            cursor.close()
            temp_conn.close()
            logging.info(f"Database '{Config.MYSQL_DB}' ensured to exist.")
        except Error as e:
            logging.error(f"Error ensuring database '{Config.MYSQL_DB}' exists: {e}")
            raise  # 向上抛出，中断 connect() 流程

    def _create_tables_if_not_exists(self):
        """
        在已连接的数据库中创建所有必要的业务表（若表不存在）。

        创建的表：
            - goods：商品信息表
                主键：goods_id（VARCHAR）
                JSON 列：scene、person、style、tags（分别存储适用场景、人群、风格、标签列表）
            - user_memory：用户记忆表
                主键：user_id（VARCHAR）
                JSON 列：preferences、forbidden_items、chat_history

        行为：
            使用 IF NOT EXISTS 语法，幂等执行，重复调用不会破坏已有数据。
            两张表在同一事务中提交，保证原子性。

        异常：
            捕获 mysql.connector.Error 并记录日志；
            finally 块确保游标无论如何都会被关闭。
        """
        # 商品信息表：存储 SKU 级别的完整商品属性
        goods_table_query = """
        CREATE TABLE IF NOT EXISTS `goods` (
            `goods_id` VARCHAR(50) PRIMARY KEY NOT NULL,
            `name` VARCHAR(255) NOT NULL,
            `category` VARCHAR(100),
            `price` DECIMAL(10, 2),
            `brand` VARCHAR(100),
            `scene` JSON,
            `person` JSON,
            `style` JSON,
            `tags` JSON,
            `feature` TEXT,
            `advantage` TEXT,
            `disadvantage` TEXT,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """

        # 用户记忆表：持久化每位用户的长期偏好、禁忌商品和对话历史
        user_memory_table_query = """
        CREATE TABLE IF NOT EXISTS `user_memory` (
            `user_id` VARCHAR(50) PRIMARY KEY NOT NULL,
            `preferences` JSON,
            `forbidden_items` JSON,
            `chat_history` JSON,
            `last_active_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(goods_table_query)
            cursor.execute(user_memory_table_query)
            # 提交 DDL 事务（MySQL 中 DDL 会隐式提交，此处显式提交以保持一致性）
            self.connection.commit()
            logging.info("All necessary tables ensured to exist.")
        except Error as e:
            logging.error(f"Error creating tables: {e}")
        finally:
            # 无论成功或失败都关闭游标，释放服务端资源
            if cursor:
                cursor.close()

    def close(self):
        """
        关闭数据库连接。

        行为：
            仅在连接对象存在且处于已连接状态时才执行关闭操作，
            避免对已关闭的连接重复调用 close() 引发异常。
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed.")

    def execute_query(self, query, params=None, fetch_type=None):
        """
        执行 SQL 语句并根据 fetch_type 返回结果。

        参数：
            query (str): 待执行的 SQL 语句，占位符使用 %s（mysql-connector 规范）。
            params (tuple | None): 与 SQL 占位符对应的参数元组，默认 None。
                                   使用参数化查询可防止 SQL 注入。
            fetch_type (str | None): 结果获取方式：
                - 'one'  → fetchone()，返回单行字典或 None
                - 'all'  → fetchall()，返回字典列表（可能为空列表）
                - None   → 不获取结果，适用于 INSERT / UPDATE / DELETE

        返回：
            dict | list[dict] | None:
                - fetch_type='one'：返回一行记录的字典，未找到时为 None。
                - fetch_type='all'：返回所有行的字典列表，无数据时为空列表。
                - fetch_type=None ：返回 None（写操作不返回数据）。
                - 发生异常时返回 None。

        行为：
            - 若当前连接已断开，自动尝试重连一次。
            - 写操作执行后自动 commit；发生异常时自动 rollback。
            - 使用 dictionary=True 游标，查询结果以列名为键的字典形式返回。
            - finally 块确保游标始终被关闭。

        异常：
            捕获 mysql.connector.Error，记录日志后返回 None，不向调用方抛出。
        """
        # 检查连接是否有效，若已断开则尝试重连（处理 MySQL 8 小时超时断连等场景）
        if not self.connection or not self.connection.is_connected():
            self.connect()  # 尝试重新建立连接
            if not self.connection or not self.connection.is_connected():
                # 重连仍然失败，无法执行查询
                logging.error("No active database connection to execute query.")
                return None

        cursor = None
        try:
            # dictionary=True：查询结果以 {列名: 值} 字典形式返回，比元组更易读
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)

            # 根据调用方需求决定如何获取结果
            if fetch_type == 'one':
                result = cursor.fetchone()       # 取单行，适用于按主键查询
            elif fetch_type == 'all':
                result = cursor.fetchall()       # 取所有行，适用于批量查询
            else:
                result = None                    # INSERT / UPDATE / DELETE 无需返回数据

            # 显式提交事务（对写操作生效；SELECT 提交无副作用）
            self.connection.commit()
            return result
        except Error as e:
            logging.error(f"Error executing query: {query} with params {params}. Error: {e}")
            # 出现异常时回滚事务，保证数据库状态一致性
            self.connection.rollback()
            return None
        finally:
            # 确保游标无论在何种情况下都会被关闭，释放服务端游标资源
            if cursor:
                cursor.close()


# 以下代码仅在直接运行 database.py 时执行，用于快速验证数据库连接和表结构是否正常
if __name__ == "__main__":
    import json

    db = Database()  # 触发单例初始化：建库 → 建连接 → 建表

    # -----------------------------------------------------------------------
    # 测试：向 goods 表插入一条样例商品
    # -----------------------------------------------------------------------
    sample_good = {
        "goods_id": "G001", "name": "简约通勤帆布包", "category": "包包",
        "price": 159.00, "brand": "平价",
        "scene": ["通勤","上学"], "person": ["女生","学生","上班族"],
        "style": ["简约","百搭"], "tags": ["性价比","送礼"],
        "feature": "大容量耐磨", "advantage": "轻便耐用", "disadvantage": "无防水"
    }
    insert_good_query = (
        "INSERT INTO goods (goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    try:
        db.execute_query(insert_good_query, (
            sample_good["goods_id"], sample_good["name"], sample_good["category"],
            sample_good["price"], sample_good["brand"],
            json.dumps(sample_good["scene"]),    # JSON 列需要先序列化为字符串
            json.dumps(sample_good["person"]),
            json.dumps(sample_good["style"]),
            json.dumps(sample_good["tags"]),
            sample_good["feature"], sample_good["advantage"], sample_good["disadvantage"],
        ))
        print("Sample good inserted.")
    except Exception as e:
        print(f"Error inserting sample good: {e}")

    # 查询刚插入的商品，验证写入是否成功
    goods = db.execute_query("SELECT * FROM goods WHERE goods_id = %s", ("G001",), fetch_type='one')
    print(f"Retrieved good: {goods}")

    db.close()
