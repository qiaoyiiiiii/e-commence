import mysql.connector
from mysql.connector import Error
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.connection = None
            cls._instance.connect()
        return cls._instance

    def connect(self):
        """ Establishes a database connection. """
        if self.connection is None or not self.connection.is_connected():
            try:
                self.connection = mysql.connector.connect(
                    host=Config.MYSQL_HOST,
                    database=Config.MYSQL_DB,
                    user=Config.MYSQL_USER,
                    password=Config.MYSQL_PASSWORD,
                    port=Config.MYSQL_PORT
                )
                if self.connection.is_connected():
                    logging.info(f"Successfully connected to MySQL database: {Config.MYSQL_DB}")
                    # Ensure the database and tables exist
                    self._create_database_if_not_exists()
                    self._create_tables_if_not_exists()
                else:
                    logging.error("Failed to connect to MySQL database.")
            except Error as e:
                logging.error(f"Error connecting to MySQL database: {e}")
                self.connection = None

    def _create_database_if_not_exists(self):
        """ Creates the database if it does not exist. """
        try:
            temp_conn = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                port=Config.MYSQL_PORT
            )
            cursor = temp_conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DB}")
            logging.info(f"Database '{Config.MYSQL_DB}' ensured to exist.")
            cursor.close()
            temp_conn.close()
            # Reconnect to the specific database
            self.connect()
        except Error as e:
            logging.error(f"Error creating database {Config.MYSQL_DB}: {e}")

    def _create_tables_if_not_exists(self):
        """ Creates necessary tables if they do not exist. """
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

        tag_library_table_query = """
        CREATE TABLE IF NOT EXISTS `tag_library` (
            `tag_id` INT AUTO_INCREMENT PRIMARY KEY,
            `tag_type` VARCHAR(100) NOT NULL,
            `tag_name` VARCHAR(100) NOT NULL UNIQUE,
            `description` TEXT,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """

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
        try:
            cursor = self.connection.cursor()
            cursor.execute(goods_table_query)
            cursor.execute(tag_library_table_query)
            cursor.execute(user_memory_table_query)
            self.connection.commit()
            logging.info("All necessary tables ensured to exist.")
        except Error as e:
            logging.error(f"Error creating tables: {e}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()

    def close(self):
        """ Closes the database connection. """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed.")

    def execute_query(self, query, params=None, fetch_type=None):
        """ Executes a query and returns results if applicable. """
        if not self.connection or not self.connection.is_connected():
            self.connect() # Attempt to reconnect
            if not self.connection or not self.connection.is_connected():
                logging.error("No active database connection to execute query.")
                return None

        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True) # Return results as dictionaries
            cursor.execute(query, params)

            if fetch_type == 'one':
                result = cursor.fetchone()
            elif fetch_type == 'all':
                result = cursor.fetchall()
            else:
                result = None # For INSERT, UPDATE, DELETE

            self.connection.commit()
            return result
        except Error as e:
            logging.error(f"Error executing query: {query} with params {params}. Error: {e}")
            self.connection.rollback()
            return None
        finally:
            if cursor:
                cursor.close()

# Example usage for testing
if __name__ == "__main__":
    db = Database()

    # Test creating tables (already done on connection)

    # Test INSERT (Goods)
    sample_good = {
        "goods_id": "G001",
        "name": "简约通勤帆布包",
        "category": "包包",
        "price": 159.00,
        "brand": "平价",
        "scene": ["通勤","上学"],
        "person": ["女生","学生","上班族"],
        "style": ["简约","百搭"],
        "tags": ["性价比","送礼"],
        "feature": "大容量耐磨",
        "advantage": "轻便耐用",
        "disadvantage": "无防水"
    }
    insert_good_query = (
        "INSERT INTO goods (goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    import json
    try:
        db.execute_query(insert_good_query, (
            sample_good["goods_id"],
            sample_good["name"],
            sample_good["category"],
            sample_good["price"],
            sample_good["brand"],
            json.dumps(sample_good["scene"]),
            json.dumps(sample_good["person"]),
            json.dumps(sample_good["style"]),
            json.dumps(sample_good["tags"]),
            sample_good["feature"],
            sample_good["advantage"],
            sample_good["disadvantage"],
        ))
        print("Sample good inserted.")
    except Exception as e:
        print(f"Error inserting sample good: {e}")

    # Test SELECT (Goods)
    select_goods_query = "SELECT * FROM goods WHERE goods_id = %s"
    goods = db.execute_query(select_goods_query, ("G001",), fetch_type='one')
    print(f"Retrieved good: {goods}")

    # Test INSERT (Tag)
    sample_tag = {
        "tag_type": "scene",
        "tag_name": "运动",
        "description": "适合运动休闲的场景"
    }
    insert_tag_query = (
        "INSERT INTO tag_library (tag_type, tag_name, description) "
        "VALUES (%s, %s, %s)"
    )
    try:
        db.execute_query(insert_tag_query, (
            sample_tag["tag_type"],
            sample_tag["tag_name"],
            sample_tag["description"],
        ))
        print("Sample tag inserted.")
    except Exception as e:
        print(f"Error inserting sample tag: {e}")

    # Test SELECT (Tag)
    select_tag_query = "SELECT * FROM tag_library WHERE tag_name = %s"
    tag = db.execute_query(select_tag_query, ("运动",), fetch_type='one')
    print(f"Retrieved tag: {tag}")

    # Test INSERT (User Memory)
    sample_user_memory = {
        "user_id": "user123",
        "preferences": {"color": "blue", "budget": "medium"},
        "forbidden_items": ["red shoes"],
        "chat_history": [{"role": "user", "message": "I need a bag"}]
    }
    insert_user_memory_query = (
        "INSERT INTO user_memory (user_id, preferences, forbidden_items, chat_history) "
        "VALUES (%s, %s, %s, %s)"
    )
    try:
        db.execute_query(insert_user_memory_query, (
            sample_user_memory["user_id"],
            json.dumps(sample_user_memory["preferences"]),
            json.dumps(sample_user_memory["forbidden_items"]),
            json.dumps(sample_user_memory["chat_history"]),
        ))
        print("Sample user memory inserted.")
    except Exception as e:
        print(f"Error inserting user memory: {e}")

    # Test SELECT (User Memory)
    select_user_memory_query = "SELECT * FROM user_memory WHERE user_id = %s"
    user_mem = db.execute_query(select_user_memory_query, ("user123",), fetch_type='one')
    print(f"Retrieved user memory: {user_mem}")

    db.close()
