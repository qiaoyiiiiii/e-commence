"""
模块：state_machine.py
职责：
    定义电商导购对话流程的状态机，用于追踪和管理多轮对话中的当前阶段。

    通过有限状态机（Finite State Machine）模型，将一次完整的导购对话分解为
    若干明确的阶段（DialogueState），并提供状态转移、历史记录和上下文管理能力。

    主要组件：
    - DialogueState（枚举类）：定义所有合法的对话阶段。
    - StateMachine（状态机类）：管理当前状态、转移逻辑、历史栈和上下文字典。

    设计说明：
    状态机当前为纯内存实现，不持久化状态。若需跨请求保持状态，
    调用方应将 current_state 和 context 序列化后存储（如存入 session 或数据库）。

依赖：
    - config.Config : 全局配置（日志级别）
    - enum.Enum     : Python 标准枚举基类
    - logging       : 标准日志记录

使用方式：
    sm = StateMachine()                                  # 初始状态为 INITIAL
    sm.transition_to(DialogueState.DEMAND_INQUIRY)       # 转移到需求问询阶段
    sm.update_context("user_intent", "寻找夏季连衣裙")    # 记录上下文信息
    print(sm.current_state)                              # 查看当前状态
    print(sm.get_state_history())                        # 查看完整状态历史
    sm.reset()                                           # 重置到初始状态
"""

import logging
from enum import Enum, auto
from typing import Dict, Any, List

from config import Config

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class DialogueState(Enum):
    """
    对话阶段枚举，定义导购对话流程中所有合法的状态节点。

    各状态说明：
    - INITIAL                  : 初始状态，等待用户第一次输入，对话尚未开始。
    - DEMAND_INQUIRY           : 需求问询阶段，Agent 已理解用户有购物意向，正在了解初步需求。
    - INFO_GATHERING           : 信息补全阶段，已有基本需求但细节不足，主动向用户询问
                                  预算、尺寸、颜色偏好等关键信息。
    - RECOMMENDATION_GENERATION: 推荐生成阶段，信息已充分，正在调用技能/RAG 生成推荐结果。
    - RECOMMENDATION_PRESENTED : 推荐已展示阶段，推荐结果已呈现给用户，等待其反馈、
                                  追问或进一步操作（如对比、加购）。
    - COMPARISON_OR_PAIRING    : 对比/搭配阶段，用户要求比较多个商品参数或寻求搭配建议。
    - SESSION_END              : 会话结束阶段，用户已完成购物意图或明确退出对话。
    """
    INITIAL = auto()                     # 初始状态：等待用户输入
    DEMAND_INQUIRY = auto()              # 需求问询：理解用户初步需求
    INFO_GATHERING = auto()              # 信息补全：收集更多细节（如预算、偏好等）
    RECOMMENDATION_GENERATION = auto()   # 推荐生成：根据收集的信息生成推荐
    RECOMMENDATION_PRESENTED = auto()    # 推荐已展示：等待用户反馈或进一步操作
    COMPARISON_OR_PAIRING = auto()       # 对比/搭配：用户要求对比商品或提供搭配建议
    SESSION_END = auto()                 # 会话结束：对话流程结束


class StateMachine:
    """
    导购对话流程状态机。

    维护当前对话阶段（DialogueState），支持状态转移、历史记录和上下文键值存储。
    所有状态变更均会记录到 _state_history 列表，便于事后审计和调试。

    设计特点：
    - 同状态转移（当前状态与目标状态相同）不会重复写入历史栈，但会记录 debug 日志。
    - _context 字典可存储任意键值对，用于在状态间传递业务信息（如用户意图、已知条件等）。
    - reset() 方法清空历史和上下文，适合在同一用户的新一轮独立对话开始时调用。

    属性：
        _current_state (DialogueState)    : 当前对话阶段（私有，通过 current_state 属性访问）。
        _state_history (List[DialogueState]): 完整的状态转移历史列表（含初始状态）。
        _context (Dict[str, Any])         : 对话过程中积累的业务上下文键值对。
    """

    def __init__(self, initial_state: DialogueState = DialogueState.INITIAL):
        """
        初始化状态机，设置初始状态并清空历史和上下文。

        参数：
            initial_state (DialogueState): 起始对话阶段，默认为 INITIAL。
                                            可传入其他状态以从中间阶段恢复（如从持久化存储还原）。
        """
        self._current_state = initial_state
        # 历史栈初始化时包含起始状态，确保历史记录完整
        self._state_history: List[DialogueState] = [initial_state]
        # 上下文字典：存储对话过程中的业务信息，如用户意图、预算、已筛选商品等
        self._context: Dict[str, Any] = {}
        logging.info(f"State machine initialized to {self._current_state.name}")

    @property
    def current_state(self) -> DialogueState:
        """
        只读属性，返回当前对话阶段。

        使用 @property 封装，防止外部直接修改 _current_state，
        所有状态变更必须通过 transition_to() 进行，确保历史记录完整性。

        返回：
            DialogueState: 当前所处的对话阶段枚举值。
        """
        return self._current_state

    def transition_to(self, new_state: DialogueState):
        """
        将状态机转移到指定的新状态，并记录到历史栈中。

        行为说明：
        - 若目标状态与当前状态不同，执行转移并追加到历史栈，同时记录 info 日志。
        - 若目标状态与当前状态相同（重复转移），不写入历史栈，仅记录 debug 日志，
          避免历史栈中出现连续重复条目。

        参数：
            new_state (DialogueState): 要转移到的目标对话阶段。
        """
        if self._current_state != new_state:
            # 真正的状态转移：更新当前状态并记录到历史
            logging.info(f"Transitioning from {self._current_state.name} to {new_state.name}")
            self._current_state = new_state
            self._state_history.append(new_state)
        else:
            # 同状态"转移"：仅记录 debug 日志，不重复写入历史
            logging.debug(f"Attempted to transition to same state: {new_state.name}")

    def get_state_history(self) -> List[DialogueState]:
        """
        获取完整的状态转移历史列表。

        返回的列表从初始状态开始，按时间顺序记录了每次实际状态转移，
        不包含同状态重复转移的记录。

        返回：
            List[DialogueState]: 按时间顺序排列的 DialogueState 枚举值列表。
        """
        return self._state_history

    def update_context(self, key: str, value: Any):
        """
        向对话上下文中添加或更新一个键值对。

        上下文字典用于在对话各阶段之间传递和积累业务信息，
        例如用户意图、已收集的约束条件、推荐结果列表等。

        参数：
            key (str) : 上下文信息的键名，建议使用描述性名称（如 "user_intent"、"budget"）。
            value (Any): 对应的值，可为任意 Python 对象（字符串、数字、列表、字典等）。
        """
        self._context[key] = value
        logging.debug(f"State machine context updated: {key} = {value}")

    def get_context(self) -> Dict[str, Any]:
        """
        获取当前完整的对话上下文字典。

        返回：
            Dict[str, Any]: 包含所有已设置的上下文键值对的字典。
                            若尚未设置任何上下文，返回空字典 {}。
        """
        return self._context

    def reset(self, initial_state: DialogueState = DialogueState.INITIAL):
        """
        重置状态机到初始状态，清空历史记录和上下文信息。

        适用场景：
        - 同一用户开始全新的独立导购对话时调用，清除上一轮会话的残留状态。
        - 测试场景中需要多次复用同一状态机实例时。

        参数：
            initial_state (DialogueState): 重置后的起始状态，默认为 INITIAL。
                                            可指定其他状态以重置到自定义起点。
        """
        self._current_state = initial_state
        # 重置历史栈，仅保留新的初始状态
        self._state_history = [initial_state]
        # 清空上下文字典，释放所有对话积累的临时业务数据
        self._context = {}
        logging.info(f"State machine reset to {self._current_state.name}")


# Example usage
if __name__ == "__main__":
    sm = StateMachine()
    print(f"Current state: {sm.current_state.name}")

    sm.transition_to(DialogueState.DEMAND_INQUIRY)
    print(f"Current state: {sm.current_state.name}")

    sm.update_context("user_intent", "寻找夏季连衣裙")
    print(f"Current context: {sm.get_context()}")

    sm.transition_to(DialogueState.INFO_GATHERING)
    print(f"Current state: {sm.current_state.name}")

    sm.transition_to(DialogueState.RECOMMENDATION_GENERATION)
    print(f"Current state: {sm.current_state.name}")

    sm.transition_to(DialogueState.SESSION_END)
    print(f"Current state: {sm.current_state.name}")

    print(f"State history: {[s.name for s in sm.get_state_history()]}")

    sm.reset()
    print(f"State after reset: {sm.current_state.name}")
