"""
模块：prompt_templates.py
职责：
    集中管理项目所有提示词模板，以及对话历史格式化工具函数。

暴露接口：
    REACT_SYSTEM_PROMPT   : ReAct Agent 系统提示词模板
    format_chat_history(messages) -> str
"""


# ReAct Agent 系统提示词
# 占位符说明：
#   {chat_history}    - 对话历史（可选，默认空字符串）
#   {input}           - 用户本轮输入
#   {agent_scratchpad}- LangChain 自动填充 Agent 的中间推理步骤
# 注意：{tools} 和 {tool_names} 不在此模板中，由 react_agent.py 在运行时手动注入
REACT_SYSTEM_PROMPT = """你是一个电商购物助手，帮助用户找到合适的商品。

历史对话：
{chat_history}

用户问题：{input}

请按以下格式思考和回答，直接调用合适的工具：
Thought: 我需要思考如何回答
Action: 工具名称
Action Input: 工具的输入
Observation: 工具返回的结果
... （可以重复 Thought/Action/Observation）
Thought: 我现在知道答案了
Final Answer: 最终回答

{agent_scratchpad}"""


def format_chat_history(messages) -> str:
    """
    将多轮对话历史列表转换为可嵌入提示词的纯文本字符串。

    支持两种消息格式：
    1. dict 格式（MCPManager 等模块传入）：
       {"role": "user"/"assistant"/"agent", "content": "..."}
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
        >>> msgs = [{"role": "user", "content": "你好"}, {"role": "agent", "content": "您好！"}]
        >>> print(format_chat_history(msgs))
        Human: 你好
        AI: 您好！
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            label = "Human" if role == "user" else "AI"
            formatted.append(f"{label}: {content}")
        else:
            # LangChain 消息对象：HumanMessage.type=="human"，AIMessage.type=="ai"
            if msg.type == "human":
                formatted.append(f"Human: {msg.content}")
            elif msg.type == "ai":
                formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)
