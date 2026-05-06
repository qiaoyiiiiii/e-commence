import logging
from typing import Dict, Callable, Any, List

from config import Config
from skills.filter_skills import FilterSkills
from skills.recommend_skills import RecommendSkills
from skills.compare_skills import CompareSkills
from skills.check_skills import CheckSkills

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class SkillRouter:
    def __init__(self):
        self._skills: Dict[str, Callable] = {}
        logging.info("SkillRouter initialized.")
        self._register_default_skills()

    def _register_default_skills(self):
        """ Registers default skills (e.g., filter skills) upon initialization. """
        filter_skills_instance = FilterSkills()
        self.register_skill("filter_goods_by_criteria", filter_skills_instance.filter_goods_by_criteria)
        self.register_skill("validate_constraints", filter_skills_instance.validate_constraints)

        recommend_skills_instance = RecommendSkills()
        self.register_skill("recommend_by_demand_matching", recommend_skills_instance.recommend_by_demand_matching)
        self.register_skill("recommend_by_personalized_preferences", recommend_skills_instance.recommend_by_personalized_preferences)

        compare_skills_instance = CompareSkills()
        self.register_skill("compare_goods_parameters", compare_skills_instance.compare_goods_parameters)

        check_skills_instance = CheckSkills()
        self.register_skill("self_reflection_check", check_skills_instance.self_reflection_check)
        # Register other skill instances here as they are created


    def register_skill(self, skill_name: str, skill_function: Callable):
        """ Registers a skill (function or LangChain Tool) with the router. """
        if skill_name in self._skills:
            logging.warning(f"Skill '{skill_name}' already registered. Overwriting.")
        self._skills[skill_name] = skill_function
        logging.info(f"Skill '{skill_name}' registered.")

    def get_skill(self, skill_name: str) -> Callable:
        """ Retrieves a registered skill function. """
        skill = self._skills.get(skill_name)
        if not skill:
            logging.error(f"Skill '{skill_name}' not found.")
            raise ValueError(f"Skill '{skill_name}' not found.")
        return skill

    def list_skills(self) -> List[str]:
        """ Returns a list of all registered skill names. """
        return list(self._skills.keys())

    def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """ Executes a registered skill with provided arguments. """
        try:
            skill_function = self.get_skill(skill_name)
            logging.info(f"Executing skill '{skill_name}' with args: {kwargs}")
            # In a real scenario, this would involve parameter validation
            # and potentially converting kwargs to tool-specific formats.
            return skill_function(**kwargs)
        except ValueError as e:
            logging.error(f"Failed to execute skill '{skill_name}': {e}")
            return None
        except Exception as e:
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
