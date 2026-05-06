from langchain.prompts import PromptTemplate

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""严格根据以下文档回答问题，找不到信息就说找不到。 

参考文档：
{context}

问题：{question}
回答：""",
    input_variables=["context", "question"]
)

# ReAct Agent Prompt Template (Placeholder - will be more complex later)
REACT_AGENT_PROMPT = """
你是一个电商导购Agent。你的目标是理解用户的需求，并使用可用的工具来帮助用户找到合适的商品。\n
{chat_history}\n
请思考你需要执行的步骤，然后执行一步：\n{agent_scratchpad}
"""

# Self-Reflection Prompt Template (Placeholder)
SELF_REFLECTION_PROMPT = """
你刚刚为用户推荐了商品。请根据用户的原始需求和当前的推荐结果，反思推荐是否合理，是否满足了所有约束。\n
用户需求: {user_query}\n推荐商品: {recommended_goods}\n
你的反思: """

# Recommendation Output Format Template (Placeholder)
RECOMMENDATION_OUTPUT_FORMAT = """
好的，为您推荐以下商品：

{recommendation_list}

如有其他需求，请告诉我。
"""

# Dialogue State Machine Prompts (Placeholder)
# For example, to ask for missing information
MISSING_INFO_PROMPT = """
我需要更多信息才能为您提供更好的推荐。您对商品的以下方面有什么偏好吗？
{missing_info_questions}
"""

# Example of a simple chat history formatter
def format_chat_history(messages):
    formatted_history = []
    for msg in messages:
        if msg.type == "human":
            formatted_history.append(f"Human: {msg.content}")
        elif msg.type == "ai":
            formatted_history.append(f"AI: {msg.content}")
    return "\n".join(formatted_history)
