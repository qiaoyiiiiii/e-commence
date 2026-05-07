"""
模块：prompt_templates.py
职责：
    集中管理本项目所有 LLM 提示词模板（Prompt Template）及对话历史格式化工具函数。
    包含以下内容：
      - RAG_PROMPT_TEMPLATE      : 检索增强生成（RAG）主提示词，供 create_retrieval_chain 使用
      - REACT_AGENT_PROMPT       : ReAct 模式导购 Agent 的提示词框架
      - SELF_REFLECTION_PROMPT   : 自省/反思提示词，用于评估推荐结果的合理性
      - RECOMMENDATION_OUTPUT_FORMAT : 推荐结果的标准输出格式模板
      - MISSING_INFO_PROMPT      : 当用户信息不足时引导追问的提示词
      - format_chat_history()    : 将对话历史列表转换为可嵌入提示词的字符串

依赖：
    - langchain.prompts.PromptTemplate：LangChain 提供的结构化提示词类

使用方式：
    from data.prompt_templates import RAG_PROMPT_TEMPLATE, format_chat_history
    history_str = format_chat_history(messages)
    # 将 history_str 作为 {chat_history} 传入 RAG_PROMPT_TEMPLATE
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# RAG 检索增强生成提示词
# ---------------------------------------------------------------------------
# 供 LangChain 的 create_retrieval_chain 调用。
# 变量说明：
#   {context}      - 由检索器（retriever）注入的相关文档片段
#   {chat_history} - 由 format_chat_history() 格式化后的多轮对话历史
#   {input}        - 用户本轮输入的问题（注意：必须用 {input}，
#                    create_retrieval_chain 固定传入该键名，不可改为 {question}）
RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""严格根据以下文档回答问题，找不到信息就说找不到。

参考文档：
{context}

{chat_history}问题：{input}
回答：""",
    input_variables=["context", "chat_history", "input"]
)

# ---------------------------------------------------------------------------
# ReAct 导购 Agent 提示词
# ---------------------------------------------------------------------------
# 供 ReAct（Reasoning + Acting）框架使用，引导 Agent 循环思考并调用工具。
# 变量说明：
#   {chat_history}     - 多轮对话历史，提供上下文
#   {agent_scratchpad} - Agent 的中间推理步骤（由框架自动填充）
REACT_AGENT_PROMPT = """
你是一个电商导购Agent。你的目标是理解用户的需求，并使用可用的工具来帮助用户找到合适的商品。\n
{chat_history}\n
请思考你需要执行的步骤，然后执行一步：\n{agent_scratchpad}
"""

# ---------------------------------------------------------------------------
# 自省反思提示词
# ---------------------------------------------------------------------------
# 在推荐结束后调用，让 LLM 评估推荐结果是否满足用户所有约束条件。
# 变量说明：
#   {user_query}        - 用户的原始需求文本
#   {recommended_goods} - 已推荐给用户的商品列表描述
SELF_REFLECTION_PROMPT = """
你刚刚为用户推荐了商品。请根据用户的原始需求和当前的推荐结果，反思推荐是否合理，是否满足了所有约束。\n
用户需求: {user_query}\n推荐商品: {recommended_goods}\n
你的反思: """

# ---------------------------------------------------------------------------
# 推荐结果输出格式模板
# ---------------------------------------------------------------------------
# 用于将推荐商品列表以统一的友好格式呈现给用户。
# 变量说明：
#   {recommendation_list} - 已格式化的商品条目文本
RECOMMENDATION_OUTPUT_FORMAT = """
好的，为您推荐以下商品：

{recommendation_list}

如有其他需求，请告诉我。
"""

# ---------------------------------------------------------------------------
# 缺少信息时的追问提示词
# ---------------------------------------------------------------------------
# 当用户需求描述不够完整时，用于引导用户补充关键偏好信息。
# 变量说明：
#   {missing_info_questions} - 需要用户回答的问题列表
MISSING_INFO_PROMPT = """
我需要更多信息才能为您提供更好的推荐。您对商品的以下方面有什么偏好吗？
{missing_info_questions}
"""


def format_chat_history(messages):
    """
    将多轮对话历史列表转换为可嵌入提示词的纯文本字符串。

    支持两种消息格式：
    1. dict 格式（MCPManager 等模块传入）：
       {"role": "user"/"assistant", "content": "..."}
    2. LangChain 消息对象格式：
       具有 .type（"human"/"ai"）和 .content 属性的对象

    参数：
        messages (list): 对话历史消息列表，每条可为 dict 或 LangChain 消息对象。

    返回：
        str: 以换行符分隔的多行字符串，格式如下：
             Human: <用户消息>
             AI: <助手消息>
             ...
             若列表为空则返回空字符串。

    示例：
        >>> msgs = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好！"}]
        >>> print(format_chat_history(msgs))
        Human: 你好
        AI: 您好！
    """
    formatted_history = []
    for msg in messages:
        if isinstance(msg, dict):
            # --- dict 格式处理 ---
            # 从字典中安全提取 role 与 content，避免 KeyError
            role = msg.get("role", "")
            content = msg.get("content", "")
            # role 为 "user" 时显示 "Human"，其余（"assistant"/"system" 等）显示 "AI"
            label = "Human" if role == "user" else "AI"
            formatted_history.append(f"{label}: {content}")
        else:
            # --- LangChain 消息对象格式处理 ---
            # LangChain 的 HumanMessage.type == "human"，AIMessage.type == "ai"
            if msg.type == "human":
                formatted_history.append(f"Human: {msg.content}")
            elif msg.type == "ai":
                formatted_history.append(f"AI: {msg.content}")
            # SystemMessage 等其他类型暂不纳入历史字符串

    # 用换行符拼接所有行，返回可直接嵌入提示词的字符串
    return "\n".join(formatted_history)
