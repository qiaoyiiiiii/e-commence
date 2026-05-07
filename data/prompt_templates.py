"""
模块：prompt_templates.py
职责：
    提供对话历史格式化工具函数，供 MCPManager 在构建 ReAct Agent 的
    chat_history 上下文时使用。

    原有的 RAG_PROMPT_TEMPLATE、REACT_AGENT_PROMPT、SELF_REFLECTION_PROMPT 等
    静态模板在切换为 ReAct Agent 架构后已不再使用，均已移除。
    若需要静态 Prompt，请在对应模块（react_agent.py / check_skills.py）中本地定义。

暴露接口：
    format_chat_history(messages) -> str
"""


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
