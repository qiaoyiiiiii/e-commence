# 电商导购Agent

本项目旨在使用 Python、LangChain、DeepSeek LLM 和 MySQL 数据库，实现一个智能电商导购 Agent。该 Agent 能够理解用户需求，进行多轮对话以提供个性化推荐，并利用检索增强生成（RAG）系统。

## 1. 项目概述

此 Agent 充当智能购物向导，具备以下能力：
*   通过自然语言理解用户需求。
*   进行多轮对话以收集必要信息。
*   使用 ReAct 风格的推理引擎做出明智决策。
*   从 MySQL 支持的 RAG 系统中检索相关产品信息。
*   利用各种技能（过滤、推荐、比较、自检）。
*   通过分层记忆系统维护用户上下文和偏好。

## 2. 核心功能

*   **RAG（检索增强生成）**: 混合检索（向量 + 关键词）和重排序，实现精准的产品搜索。
*   **ReAct（推理 + 行动）**: 未来将集成，用于高级决策和工具使用。
*   **分层记忆**: 短期记忆（会话历史），长期记忆（用户偏好、MySQL 中的历史交互）。
*   **MCP（多轮对话控制协议）**: 管理对话流程、上下文和信息收集。
*   **技能框架**: 模块化和可扩展的工具集，用于特定任务（过滤、推荐、比较、自省）。
*   **DeepSeek LLM 集成**: 利用 DeepSeek 模型进行自然语言理解和生成。
*   **MySQL 存储**: 所有持久化数据（产品目录、标签、用户记忆）都存储在 MySQL 数据库中。

## 3. 项目结构

```
ecommerce_shopping_agent/
├── main.py                 # 项目入口，命令行交互
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖项
├── .env                    # 环境变量（本地设置）
├── database.py             # MySQL 数据库连接和 CRUD 操作
├── agent_core/             # Agent 核心模块
│   ├── __init__.py
│   ├── react_agent.py      # DeepSeek LLM 集成（将演变为 ReAct 引擎）
│   ├── memory_manager.py   # 分层记忆管理（长期记忆在 MySQL 中）
│   ├── mcp_manager.py      # 多轮对话控制
│   ├── state_machine.py    # 对话状态机
│   └── skill_router.py     # 技能路由和执行
├── rag_module/             # RAG 检索模块
│   ├── __init__.py
│   ├── data_processor.py   # 从 MySQL 加载产品数据并转换为 LangChain Document
│   ├── hybrid_retriever.py # 多路径检索（向量 + 关键词）
│   ├── reranker.py         # 结果重排序
│   └── vector_store.py     # ChromaDB 向量存储管理
├── skills/                 # 技能工具集
│   ├── __init__.py
│   ├── filter_skills.py    # 价格/标签/约束过滤
│   ├── recommend_skills.py # 匹配/个性化推荐
│   ├── compare_skills.py   # 商品比较
│   └── check_skills.py     # 自省/验证技能
├── data/                   # 数据相关文件
│   └── prompt_templates.py # Prompt 模板
└──  user_memory/            # 占位符（长期记忆在 MySQL 中）
```

## 4. 设置说明

### 先决条件

*   Python 3.10+
*   MySQL 服务器（例如：XAMPP、Docker 或独立安装）
*   DeepSeek API 密钥

### 虚拟环境

```bash
python -m venv .venv
# 在 Windows 上
.venv\Scripts\activate
# 在 macOS/Linux 上
source .venv/bin/activate
```

### 依赖项

安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

### 环境变量 (.env)

在项目根目录创建 `.env` 文件，并填写您的凭据和配置。请确保将占位符替换为您的实际值。

```env
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"

MYSQL_HOST="localhost"
MYSQL_USER="root"
MYSQL_PASSWORD="YOUR_MYSQL_ROOT_PASSWORD" # 或者专用用户的密码
MYSQL_DB="ecommerce_agent"
MYSQL_PORT=3306

# Langchain 相关（可选，用于跟踪）
LANGCHAIN_TRACING_V2="false"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="E-commerce Shopping Agent"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Ollama (如果您仍想将其用于其他目的或本地测试)
# OLLAMA_MODEL="qwen2.5:7b"
# OLLAMA_HOST="http://127.0.0.1:11434"

# RAG 配置
EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL_DEVICE="cpu" # 如果没有 CUDA GPU，则为 "cpu"
RERANKER_MODEL_NAME="BAAI/bge-reranker-large"
RERANKER_MODEL_DEVICE="cpu" # 或 "cpu"
RETRIEVER_K=20
RERANKER_TOP_N=5

# Agent 特有配置
RECOMMENDATION_COUNT=3
SELF_REFLECTION_ENABLED="True"
RERANKING_ENABLED="True"

# ChromaDB 持久化路径
CHROMA_DB_PATH="./chroma_db"

# 日志
LOG_LEVEL="INFO"
DEBUG_MODE="False"
```

### MySQL 数据库设置

确保您的 MySQL 服务器正在运行。当应用程序连接时，`database.py` 脚本将尝试创建 `ecommerce_agent` 数据库和必要的表（`goods`、`tag_library`、`user_memory`）（如果它们不存在）。您需要在 `.env` 文件中提供具有创建数据库和表权限的相应 MySQL 用户凭据。

或者，您可以使用 MySQL 客户端（例如：MySQL Workbench、DBeaver 或命令行）手动创建数据库和表：

1.  **连接到 MySQL** 作为具有足够权限的用户（例如 `root`）。

2.  **创建数据库**：
    ```sql
    CREATE DATABASE IF NOT EXISTS ecommerce_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    USE ecommerce_agent;
    ```

3.  **创建表**：
    ```sql
    CREATE TABLE `goods` (
        `goods_id` VARCHAR(50) PRIMARY KEY NOT NULL,
        `name` VARCHAR(255) NOT NULL,
        `category` VARCHAR(100),
        `price` DECIMAL(10, 2),
        `brand` VARCHAR(100),
        `scene` JSON, -- 存储为 JSON 字符串数组
        `person` JSON, -- 存储为 JSON 字符串数组
        `style` JSON, -- 存储为 JSON 字符串数组
        `tags` JSON, -- 存储为 JSON 字符串数组
        `feature` TEXT,
        `advantage` TEXT,
        `disadvantage` TEXT,
        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );

    CREATE TABLE `tag_library` (
        `tag_id` INT AUTO_INCREMENT PRIMARY KEY,
        `tag_type` VARCHAR(100) NOT NULL, -- 例如: "scene", "person", "style", "budget", "usage"
        `tag_name` VARCHAR(100) NOT NULL UNIQUE,
        `description` TEXT,
        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );

    CREATE TABLE `user_memory` (
        `user_id` VARCHAR(50) PRIMARY KEY NOT NULL,
        `preferences` JSON, -- 用户偏好（例如：预算、风格、品牌）
        `forbidden_items` JSON, -- 用户不喜欢的商品或类别
        `chat_history` JSON, -- 长期上下文的聊天历史摘要或完整记录
        `last_active_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ```

### 插入示例数据

为了使 RAG 系统和技能正常工作，您需要向 `goods` 和 `tag_library` 表填充一些数据。以下是示例 `INSERT` 语句。您可以使用 MySQL 客户端执行这些语句。

**示例 `goods` 数据：**

```sql
INSERT INTO `goods` (goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage)
VALUES
('G001', '简约通勤帆布包', '包包', 159.00, '平价', '["通勤","上学"]', '["女生","学生","上班族"]', '["简约","百搭"]', '["性价比","送礼"]', '大容量耐磨', '轻便耐用', '无防水'),
('G002', '时尚潮流双肩包', '包包', 299.00, '潮流品牌', '["休闲","旅行"]', '["学生","青年"]', '["时尚","街头"]', '["多功能","防水"]', '多隔层设计', '时尚耐用，适合旅行', '自重稍大'),
('G003', '商务真皮公文包', '包包', 899.00, '高端品牌', '["商务","会议"]', '["上班族","商务人士"]', '["商务","精英"]', '["真皮","品质"]', '多功能分区，可放电脑', '质感高级，彰显品味', '价格较高'),
('G004', '夏季清凉雪纺连衣裙', '服装', 188.00, '快时尚', '["日常","约会"]', '["女生","青年"]', '["甜美","休闲"]', '["透气","舒适"]', '轻薄面料，不贴身', '穿着舒适，款式时尚', '易皱，需手洗'),
('G005', '户外运动跑步鞋', '鞋子', 450.00, '专业运动', '["运动","户外"]', '["运动爱好者","男士"]', '["功能性","科技感"]', '["减震","防滑"]', '专业缓震技术', '保护性强，抓地力好', '日常搭配受限');
```

**示例 `tag_library` 数据：**

```sql
INSERT INTO `tag_library` (tag_type, tag_name, description)
VALUES
('scene', '通勤', '适合日常上下班或上学的场景'),
('scene', '上学', '适合学生日常使用的场景'),
('scene', '休闲', '适合放松休闲的日常场合'),
('scene', '旅行', '适合出门旅行或短期出差的场景'),
('scene', '商务', '适合正式商务场合'),
('scene', '约会', '适合情侣约会或朋友聚会'),
('scene', '运动', '适合体育锻炼和户外活动'),
('person', '女生', '适合女性用户'),
('person', '学生', '适合学生群体'),
('person', '上班族', '适合办公室工作人员'),
('person', '青年', '适合年轻用户'),
('person', '商务人士', '适合从事商务活动的人群'),
('person', '运动爱好者', '适合热爱运动的人群'),
('person', '男士', '适合男性用户'),
('style', '简约', '设计简洁，不花哨'),
('style', '百搭', '容易搭配各种服饰'),
('style', '时尚', '符合当前流行趋势'),
('style', '街头', '具有街头潮流风格'),
('style', '商务', '正式、专业的风格'),
('style', '精英', '高端、品质的风格'),
('style', '甜美', '可爱、柔和的风格'),
('style', '休闲', '轻松、随意的风格'),
('style', '潮流', '引领时尚前沿'),
('style', '功能性', '注重实用和特定功能'),
('style', '科技感', '具有高科技元素的风格'),
('tags', '性价比', '价格合理，性能优越'),
('tags', '送礼', '适合作为礼物赠送'),
('tags', '多功能', '具有多种用途或功能'),
('tags', '防水', '具备防水特性'),
('tags', '真皮', '采用真皮材质'),
('tags', '品质', '产品质量上乘'),
('tags', '透气', '材质透气性好'),
('tags', '舒适', '穿着或使用感受良好'),
('tags', '减震', '具备减震功能'),
('tags', '防滑', '具备防滑功能');
```

## 5. 如何运行 Agent

1.  **激活您的虚拟环境**（如果尚未激活）：
    ```bash
    # 在 Windows 上
    .venv\Scripts\activate
    # 在 macOS/Linux 上
    source .venv/bin/activate
    ```

2.  **运行主应用程序**：
    ```bash
    python main.py --user_id your_unique_user_id
    ```
    将 `your_unique_user_id` 替换为您希望用于会话的任何标识符。此 ID 将用于在数据库中管理您的长期记忆。

3.  在命令行中**与 Agent 交互**。输入您的查询并按 Enter 键。输入 `exit` 或 `quit` 结束会话。

    交互示例：
    ```
    您有什么需求？ 给我推荐一款适合上班族用的简约风格包包
    ```

## 6. 未来增强

*   实现完整的 ReAct Agent，支持动态工具调用。
*   更高级的多轮对话管理（缺失信息检测、指代消解）。
*   更复杂的个性化推荐逻辑。
*   在 `CompareSkills` 中集成 LLM 进行优缺点总结。
*   实时日志记录到文件。
*   基于 Web 的 UI，实现更丰富的交互。
*   更强大的错误处理和输入验证。
