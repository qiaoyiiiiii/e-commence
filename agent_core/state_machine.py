import logging
from enum import Enum, auto
from typing import Dict, Any

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class DialogueState(Enum):
    INITIAL = auto()              # 初始状态：等待用户输入
    DEMAND_INQUIRY = auto()       # 需求问询：理解用户初步需求
    INFO_GATHERING = auto()       # 信息补全：收集更多细节（如预算、偏好等）
    RECOMMENDATION_GENERATION = auto() # 推荐生成：根据收集的信息生成推荐
    RECOMMENDATION_PRESENTED = auto() # 推荐已展示：等待用户反馈或进一步操作
    COMPARISON_OR_PAIRING = auto() # 对比/搭配：用户要求对比商品或提供搭配建议
    SESSION_END = auto()          # 会话结束：对话流程结束

class StateMachine:
    def __init__(self, initial_state: DialogueState = DialogueState.INITIAL):
        self._current_state = initial_state
        self._state_history: List[DialogueState] = [initial_state]
        self._context: Dict[str, Any] = {}
        logging.info(f"State machine initialized to {self._current_state.name}")

    @property
    def current_state(self) -> DialogueState:
        """ Returns the current state of the dialogue. """
        return self._current_state

    def transition_to(self, new_state: DialogueState):
        """ Transitions the state machine to a new state. """
        if self._current_state != new_state:
            logging.info(f"Transitioning from {self._current_state.name} to {new_state.name}")
            self._current_state = new_state
            self._state_history.append(new_state)
        else:
            logging.debug(f"Attempted to transition to same state: {new_state.name}")

    def get_state_history(self) -> List[DialogueState]:
        """ Returns the history of states. """
        return self._state_history

    def update_context(self, key: str, value: Any):
        """ Updates the state machine's context. """
        self._context[key] = value
        logging.debug(f"State machine context updated: {key} = {value}")

    def get_context(self) -> Dict[str, Any]:
        """ Returns the current context of the state machine. """
        return self._context

    def reset(self, initial_state: DialogueState = DialogueState.INITIAL):
        """ Resets the state machine to its initial state. """
        self._current_state = initial_state
        self._state_history = [initial_state]
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
