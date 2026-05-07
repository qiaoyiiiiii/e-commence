# 电商导购 Agent

基于 **ReAct + RAG** 架构的智能电商导购助手，使用 DeepSeek LLM、LangChain、ChromaDB 和 MySQL 构建。Agent 能够理解自然语言需求，通过工具自主决策，进行多轮对话并提供个性化商品推荐。

---

## 架构概览

```
用户输入
    ↓
main.py  →  MCPManager.process_user_input()
                ↓  记忆写入 + 历史摘要 + 偏好增强
            ReActAgentEngine.run()
                ↓  Thought → Action → Observation 循环
            tools/tool_loader  （Agent 接口适配层）
                ↓  按意图自动选择工具
    ┌───────────┬───────────┬───────────┬──────────────┐
recommend   filter    compare   self_reflection
    ↓           ↓           ↓           ↓
skills/  →  MySQL / ChromaDB(向量+BM25) / DeepSeek LLM
```

**分层设计：**

| 层 | 目录/文件 | 职责 |
|---|---|---|
| 入口 & CLI | `main.py` | 交互循环，生命周期管理 |
| 对话协调 | `agent_core/mcp_manager.py` | 记忆管理、偏好增强、Agent 调度 |
| 推理引擎 | `agent_core/react_agent.py` | ReAct 循环（Thought/Action/Observation） |
| 工具适配 | `tools/` | 将 skills 包装为 Agent 可调用的 Tool，处理字符串 I/O |
| 业务技能 | `skills/` | 推荐、过滤、对比、自省的具体实现 |
| 检索 | `rag_module/` | BM25 + 向量混合检索，Cross-Encoder 重排序 |
| 记忆 | `agent_core/memory_manager.py` | 短期（会话列表）+ 长期（MySQL）+ 滚动摘要压缩 |

---

## 核心能力

- **ReAct 自主决策**：Agent 根据用户意图自动选择工具（推荐/过滤/对比/自省），无需硬编码分支
- **混合检索（Hybrid RAG）**：BM25 关键词 + 向量语义双路召回，RRF 融合排序，Cross-Encoder 精排
- **分层记忆**：短期会话历史 + 长期偏好/禁忌持久化到 MySQL，超出阈值时 LLM 自动生成滚动摘要
- **个性化增强**：用户偏好和禁忌商品在每次查询前自动拼入，引导检索和推荐方向
- **自我反思**：推荐完成后可调用 `self_reflection_check`，由 LLM 评估推荐质量

---

## 项目结构

```
├── main.py                      # 入口：CLI 交互循环
├── config.py                    # 全局配置（读取 .env）
├── database.py                  # MySQL 单例连接，自动建库建表
├── requirements.txt
├── .env                         # 本地凭据（不提交到 Git）
│
├── agent_core/
│   ├── react_agent.py           # ReActAgentEngine + DeepSeekLLM
│   ├── mcp_manager.py           # 对话协调：记忆 + 偏好 + Agent 调度
│   ├── memory_manager.py        # 短期/长期记忆读写、LLM 摘要压缩
│   └── skill_router.py          # 技能注册与执行路由
│
├── tools/                       # Agent 接口适配层（skills → LangChain Tool）
│   ├── tool_loader.py           # 聚合所有工具，暴露 get_all_tools(user_id)
│   ├── recommend_tools.py       # 封装推荐类技能
│   ├── filter_tools.py          # 封装过滤/校验类技能
│   ├── compare_tools.py         # 封装对比类技能
│   └── check_tools.py           # 封装自省类技能
│
├── skills/                      # 业务逻辑层
│   ├── recommend_skills.py      # 需求匹配推荐 + 个性化推荐（RAG）
│   ├── filter_skills.py         # 多条件 SQL 过滤 + 约束校验
│   ├── compare_skills.py        # 商品参数对比
│   └── check_skills.py          # LLM 自我反思评估
│
├── rag_module/
│   ├── hybrid_retriever.py      # BM25 + 向量双路 + RRF 融合 + Reranker
│   ├── vector_store.py          # ChromaDB 向量库（自动建库/增量同步）
│   ├── reranker.py              # Cross-Encoder 重排序
│   └── data_processor.py        # MySQL → LangChain Document 转换
│
└── data/
    ├── prompt_templates.py      # format_chat_history() 工具函数
    ├── goods_data.json          # 初始商品种子数据
    └── tag_library.json         # 初始标签库种子数据
```

---

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 `.env`

在项目根目录创建 `.env` 文件：

```env
# DeepSeek API
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_MODEL_NAME="deepseek-chat"
LLM_TEMPERATURE=0.0

# MySQL
MYSQL_HOST="localhost"
MYSQL_USER="root"
MYSQL_PASSWORD="your_password"
MYSQL_DB="ecommerce_agent"
MYSQL_PORT=3306

# ChromaDB
CHROMA_DB_PATH="./chroma_db"
COLLECTION_NAME="ecommerce_goods"

# 嵌入模型（BGE，需提前下载或联网拉取）
EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL_DEVICE="cpu"

# 重排序模型
RERANKER_MODEL_NAME="BAAI/bge-reranker-large"
RERANKER_MODEL_DEVICE="cpu"
RERANKER_TOP_N=5

# 检索配置
RETRIEVER_K=20
BM25_ENABLED="True"
ENSEMBLE_VECTOR_WEIGHT="0.5"
RERANKING_ENABLED="True"

# Agent 行为
RECOMMENDATION_COUNT=3
SELF_REFLECTION_ENABLED="True"

# 记忆压缩
MEMORY_COMPRESSION_THRESHOLD=10
MEMORY_COMPRESSION_KEEP_RECENT=4

# 日志
LOG_LEVEL="INFO"
DEBUG_MODE="False"
```

### 3. 数据库初始化

启动 MySQL 后直接运行项目，`database.py` 会自动创建数据库和以下三张表：

```sql
-- 商品表
goods (goods_id, name, category, price, brand,
       scene JSON, person JSON, style JSON, tags JSON,
       feature, advantage, disadvantage)

-- 标签库
tag_library (tag_id, tag_type, tag_name, description)

-- 用户记忆
user_memory (user_id, preferences JSON, forbidden_items JSON,
             chat_history TEXT, last_active_at, created_at)
```

**插入示例商品数据：**

```sql
INSERT INTO goods (goods_id, name, category, price, brand, scene, person, style, tags, feature, advantage, disadvantage)
VALUES
('G001','简约通勤帆布包','包包',159.00,'平价','["通勤","上学"]','["女生","学生","上班族"]','["简约","百搭"]','["性价比","送礼"]','大容量耐磨','轻便耐用','无防水'),
('G002','时尚潮流双肩包','包包',299.00,'潮流品牌','["休闲","旅行"]','["学生","青年"]','["时尚","街头"]','["多功能","防水"]','多隔层设计','时尚耐用，适合旅行','自重稍大'),
('G003','商务真皮公文包','包包',899.00,'高端品牌','["商务","会议"]','["上班族","商务人士"]','["商务","精英"]','["真皮","品质"]','多功能分区，可放电脑','质感高级，彰显品味','价格较高'),
('G004','夏季清凉雪纺连衣裙','服装',188.00,'快时尚','["日常","约会"]','["女生","青年"]','["甜美","休闲"]','["透气","舒适"]','轻薄面料，不贴身','穿着舒适，款式时尚','易皱，需手洗'),
('G005','户外运动跑步鞋','鞋子',450.00,'专业运动','["运动","户外"]','["运动爱好者","男士"]','["功能性","科技感"]','["减震","防滑"]','专业缓震技术','保护性强，抓地力好','日常搭配受限');
```

### 4. 运行

```bash
python main.py
# 指定用户 ID（用于隔离记忆数据）
python main.py --user_id alice
```

**交互示例：**

```
您有什么需求？ 给我推荐一款适合上班族用的简约风格包包
您有什么需求？ 帮我比较一下 G001 和 G003
您有什么需求？ 价格在 200 元以内的包包有哪些
您有什么需求？ exit
```

---

## 技术栈

| 组件 | 技术 |
|---|---|
| LLM | DeepSeek Chat（via `langchain-deepseek`） |
| 向量嵌入 | BAAI/bge-large-zh-v1.5（HuggingFace） |
| 重排序 | BAAI/bge-reranker-large（Cross-Encoder） |
| 向量数据库 | ChromaDB（持久化） |
| 关键词检索 | rank_bm25（字符级中文分词） |
| 检索融合 | EnsembleRetriever（RRF，c=60） |
| Agent 框架 | LangChain ReAct（create_react_agent + AgentExecutor） |
| 关系数据库 | MySQL 8.0+（商品、标签、用户记忆） |
| 配置管理 | python-dotenv |
