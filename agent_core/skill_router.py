"""
模块：skill_router.py
职责：
    技能路由器，负责集中注册、管理和调度智能体所有可用技能（Skill）。

    技能是对特定业务能力的封装（如商品过滤、个性化推荐、商品对比、自我反思检查等），
    每个技能对应一个可调用的 Python 函数（Callable）。
    SkillRouter 将所有技能统一管理，供 ReActAgentEngine 在构建工具集时批量获取，
    也可在其他模块中直接按名称执行指定技能。

    默认注册技能（在 __init__ 中自动完成）：
    - filter_goods_by_criteria          : 按条件过滤商品（FilterSkills）
    - validate_constraints              : 校验用户约束条件（FilterSkills）
    - recommend_by_demand_matching      : 基于需求匹配推荐商品（RecommendSkills）
    - recommend_by_personalized_preferences : 基于个性化偏好推荐（RecommendSkills）
    - compare_goods_parameters          : 对比商品参数（CompareSkills）
    - self_reflection_check             : Agent 自我反思检查（CheckSkills）

依赖：
    - config.Config                   : 全局配置（日志级别等）
    - skills.filter_skills.FilterSkills     : 商品过滤技能类
    - skills.recommend_skills.RecommendSkills : 推荐技能类
    - skills.compare_skills.CompareSkills   : 对比技能类
    - skills.check_skills.CheckSkills       : 检查技能类

使用方式：
    router = SkillRouter()
    # 查看所有已注册技能名称
    print(router.list_skills())
    # 执行指定技能
    result = router.execute_skill("recommend_by_demand_matching", user_query="我想买一件蓝色连衣裙")
    # 手动注册额外技能
    router.register_skill("my_custom_skill", my_function)
"""

import logging
from typing import Dict, Callable, Any, List

from config import Config
from skills.filter_skills import FilterSkills
from skills.recommend_skills import RecommendSkills
from skills.compare_skills import CompareSkills
from skills.check_skills import CheckSkills

# 按照全局配置初始化日志，格式包含时间戳、日志级别和消息内容
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class SkillRouter:
    """
    技能路由器，统一管理所有可用技能的注册、查询与执行。

    以字典 _skills 为核心存储结构，键为技能名称（str），值为对应的可调用函数（Callable）。
    初始化时自动注册所有默认技能，也支持运行时动态添加新技能。

    典型使用场景：
    1. ReActAgentEngine 在构建工具列表时，调用 list_skills() 遍历所有技能，
       将每个技能封装为 LangChain Tool 注入 Agent。
    2. 直接调用 execute_skill() 执行某项具体业务逻辑。

    属性：
        _skills (Dict[str, Callable]): 技能名称到技能函数的映射字典（私有）。
    """

    def __init__(self):
        """
        初始化 SkillRouter，创建空的技能字典，并立即注册所有默认技能。
        """
        # 使用字典存储技能映射，键为技能名，值为可调用函数
        self._skills: Dict[str, Callable] = {}
        logging.info("SkillRouter initialized.")
        # 初始化时立即注册系统内所有默认技能
        self._register_default_skills()

    def _register_default_skills(self):
        """
        注册系统内置的所有默认技能。

        按技能类分组实例化，并将各实例的方法逐一注册到路由表中。
        新增技能类时，在此方法中添加对应的实例化和注册调用即可。

        注册的技能：
        - FilterSkills    : filter_goods_by_criteria（按条件过滤）、validate_constraints（约束校验）
        - RecommendSkills : recommend_by_demand_matching（需求匹配推荐）、
                            recommend_by_personalized_preferences（个性化推荐）
        - CompareSkills   : compare_goods_parameters（商品参数对比）
        - CheckSkills     : self_reflection_check（Agent 自我反思检查）
        """
        # 注册过滤类技能
        filter_skills_instance = FilterSkills()
        self.register_skill("filter_goods_by_criteria", filter_skills_instance.filter_goods_by_criteria)
        self.register_skill("validate_constraints", filter_skills_instance.validate_constraints)

        # 注册推荐类技能
        recommend_skills_instance = RecommendSkills()
        self.register_skill("recommend_by_demand_matching", recommend_skills_instance.recommend_by_demand_matching)
        self.register_skill("recommend_by_personalized_preferences", recommend_skills_instance.recommend_by_personalized_preferences)

        # 注册对比类技能
        compare_skills_instance = CompareSkills()
        self.register_skill("compare_goods_parameters", compare_skills_instance.compare_goods_parameters)

        # 注册检查类技能（Agent 自我反思，用于校验推荐结果的合理性）
        check_skills_instance = CheckSkills()
        self.register_skill("self_reflection_check", check_skills_instance.self_reflection_check)
        # 如需添加更多技能类，在此处继续实例化和注册

    def register_skill(self, skill_name: str, skill_function: Callable):
        """
        向路由表注册一个技能函数。

        若同名技能已存在，会覆盖原有函数并发出警告日志，以便开发者感知潜在的重复注册问题。

        参数：
            skill_name (str)      : 技能的唯一名称标识符，建议使用下划线命名（snake_case）。
            skill_function (Callable): 技能对应的可调用函数或方法，
                                       调用时会接收 **kwargs 形式的参数。
        """
        if skill_name in self._skills:
            # 发出警告，提示开发者同名技能将被覆盖
            logging.warning(f"Skill '{skill_name}' already registered. Overwriting.")
        self._skills[skill_name] = skill_function
        logging.info(f"Skill '{skill_name}' registered.")

    def get_skill(self, skill_name: str) -> Callable:
        """
        按名称获取已注册的技能函数。

        参数：
            skill_name (str): 要获取的技能名称。

        返回：
            Callable: 对应的技能函数。

        异常：
            ValueError: 若指定名称的技能未注册，则抛出此异常，并记录错误日志。
        """
        skill = self._skills.get(skill_name)
        if not skill:
            # 技能不存在时记录错误并抛出异常，便于调用方感知和处理
            logging.error(f"Skill '{skill_name}' not found.")
            raise ValueError(f"Skill '{skill_name}' not found.")
        return skill

    def list_skills(self) -> List[str]:
        """
        返回所有已注册技能的名称列表。

        返回：
            List[str]: 技能名称列表，顺序与注册顺序一致（Python 3.7+ 字典保持插入顺序）。

        典型用途：
            ReActAgentEngine 调用此方法遍历技能名称，将每个技能封装为 LangChain Tool。
        """
        return list(self._skills.keys())

    def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """
        按名称执行指定技能，并传入关键字参数。

        内部流程：
            1. 调用 get_skill() 查找技能函数，若不存在则捕获 ValueError 返回 None。
            2. 记录执行日志（技能名称和参数）。
            3. 调用技能函数，将 kwargs 解包后传入。
            4. 捕获执行过程中的所有异常，避免单个技能失败影响主流程。

        参数：
            skill_name (str): 要执行的技能名称。
            **kwargs         : 传递给技能函数的关键字参数，如 user_query="..."。

        返回：
            Any : 技能函数的返回值。
                  若技能不存在或执行过程中发生异常，返回 None，并记录错误日志。
        """
        try:
            skill_function = self.get_skill(skill_name)
            logging.info(f"Executing skill '{skill_name}' with args: {kwargs}")
            # 将 kwargs 解包后传入技能函数，支持任意参数组合
            return skill_function(**kwargs)
        except ValueError as e:
            # 技能不存在时记录错误，返回 None
            logging.error(f"Failed to execute skill '{skill_name}': {e}")
            return None
        except Exception as e:
            # 捕获技能执行过程中的所有未预期异常，防止崩溃传播到上层
            logging.error(f"An unexpected error occurred during skill '{skill_name}' execution: {e}")
            return None

    # Future methods:
    # def validate_skill_parameters(self, skill_name: str, params: Dict[str, Any]) -> bool:
    #     """ Validates if the given parameters match the skill's requirements. """
    #     pass
    #
    # def execute_chained_skills(self, skill_sequence: List[Dict[str, Any]], initial_input: Any) -> Any:
    #     """ Executes a sequence of skills, passing output of one as input to the next. """
    #     pass


# Example usage
if __name__ == "__main__":
    router = SkillRouter()

    # Define some dummy skills
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    def add(a: int, b: int) -> int:
        return a + b

    router.register_skill("greet_user", greet)
    router.register_skill("add_numbers", add)

    print(f"Registered skills: {router.list_skills()}")

    # Execute skills
    print(router.execute_skill("greet_user", name="Alice"))
    print(router.execute_skill("add_numbers", a=5, b=3))

    # Test non-existent skill
    print(router.execute_skill("non_existent_skill"))
