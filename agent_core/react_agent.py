import logging
from typing import List, Optional

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOpenAI

from config import Config
from rag_module.hybrid_retriever import HybridRetrieverManager
from data.prompt_templates import REACT_AGENT_PROMPT

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# ReAct Prompt Template (Standard)
REACT_SYSTEM_PROMPT = """
你是一个智能电商导购助手。你的目标是帮助用户找到合适的商品。
你可以使用以下工具：

{tools}

使用以下格式进行思考和行动：

Question: 用户输入的问题
Thought: 你应该始终思考下一步该做什么
Action: 应该采取的行动，必须是 [{tool_names}] 中的一个
Action Input: 行动的输入参数
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终回答

开始！

Question: {input}
Thought:{agent_scratchpad}
"""

class ReActAgentEngine:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.agent_executor = self._create_agent_executor()
        logging.info(f"ReActAgentEngine initialized for user {self.user_id}")

    def _initialize_llm(self):
        """ Initializes the DeepSeek LLM instance using OpenAI compatible interface. """
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        
        try:
            # DeepSeek is compatible with OpenAI API
            llm = ChatOpenAI(
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=Config.LLM_TEMPERATURE,
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
            )
            logging.info(f"DeepSeek LLM initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM: {e}")
            raise

    def _initialize_tools(self) -> List[Tool]:
        """ Initializes tools (Skills and RAG) for the Agent. """
        # Move import inside the method to avoid circular dependency
        from agent_core.skill_router import SkillRouter
        
        tools = []
        
        # 1. Initialize Skill Router and register skills as tools
        skill_router = SkillRouter()
        registered_skills = skill_router.list_skills()
        
        for skill_name in registered_skills:
            def skill_wrapper(query, name=skill_name):
                # Note: In a real scenario, we might need to parse 'query' into specific kwargs for the skill
                # For now, we pass a generic query or empty dict if the skill doesn't need complex parsing
                # A more advanced implementation would use an LLM to extract parameters from the query for each skill
                try:
                    # Simple heuristic: if skill needs specific args, this might fail or need refinement
                    # For demonstration, we assume skills can handle a generic 'query' or we pass empty
                    # Ideally, each tool should have a defined argument schema
                    if name == "filter_goods_by_criteria":
                         # Example of mapping natural language to structured args would go here
                         # For now, just returning a placeholder or executing with defaults
                         return skill_router.execute_skill(name) 
                    else:
                        return skill_router.execute_skill(name)
                except Exception as e:
                    return f"Error executing skill {name}: {str(e)}"

            tool = Tool(
                name=skill_name,
                func=skill_wrapper,
                description=f"Useful for when you need to {skill_name}. Input should be a relevant query or parameters."
            )
            tools.append(tool)

        # 2. Initialize RAG as a tool
        try:
            hybrid_retriever_manager = HybridRetrieverManager()
            retriever = hybrid_retriever_manager.get_retriever()
            
            def rag_search(query: str) -> str:
                """Searches for products based on the user's query using RAG."""
                docs = retriever.invoke(query)
                if not docs:
                    return "No relevant products found."
                
                formatted_docs = []
                for doc in docs:
                    name = doc.metadata.get('name', 'Unknown')
                    price = doc.metadata.get('price', 'N/A')
                    feature = doc.metadata.get('feature', '')
                    formatted_docs.append(f"Product: {name}, Price: {price}, Feature: {feature}")
                
                return "\n".join(formatted_docs)

            rag_tool = Tool(
                name="product_search",
                func=rag_search,
                description="Useful for when you need to search for products based on natural language descriptions. Input should be a detailed product description or requirement."
            )
            tools.append(rag_tool)
        except Exception as e:
            logging.warning(f"Failed to initialize RAG tool: {e}. RAG functionality will be unavailable.")

        return tools

    def _create_agent_executor(self) -> AgentExecutor:
        """ Creates the ReAct Agent Executor. """
        prompt = PromptTemplate.from_template(REACT_SYSTEM_PROMPT)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=Config.DEBUG_MODE,
            max_iterations=5, # Prevent infinite loops
            handle_parsing_errors=True
        )
        return agent_executor

    def run(self, user_input: str) -> str:
        """ Runs the ReAct Agent with the given user input. """
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "Sorry, I couldn't generate a response.")
        except Exception as e:
            logging.error(f"Error running ReAct Agent: {e}")
            return f"Sorry, an error occurred while processing your request: {str(e)}"

# Legacy class name for backward compatibility if needed, but recommended to use ReActAgentEngine
class DeepSeekLLM:
    """ Wrapper for simple LLM calls without ReAct. Decoupled from ReActAgentEngine to avoid circular imports. """
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """ Initializes the DeepSeek LLM instance using OpenAI compatible interface. """
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        
        try:
            # DeepSeek is compatible with OpenAI API
            llm = ChatOpenAI(
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=Config.LLM_TEMPERATURE,
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
            )
            logging.info(f"DeepSeek LLM (Direct) initialized with model: {Config.DEEPSEEK_MODEL_NAME}")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize DeepSeek LLM (Direct): {e}")
            raise

    def get_llm(self):
        return self.llm

    def invoke(self, prompt: str):
        """ Direct invocation without ReAct loop (for simple tasks or legacy code). """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logging.error(f"Error invoking DeepSeek LLM directly: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Make sure to set DEEPSEEK_API_KEY in your .env file before running this example
    try:
        agent_engine = ReActAgentEngine(user_id="test_user")
        
        test_query = "我想找一个适合通勤的包包，价格不要太贵，最好有性价比。"
        print(f"\nUser Query: {test_query}")
        print("-" * 50)
        
        response = agent_engine.run(test_query)
        print(f"Agent Response:\n{response}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")