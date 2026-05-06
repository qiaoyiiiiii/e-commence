from langchain_deepseek import ChatDeepSeek
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekLLM:
    def __init__(self):
        if not Config.DEEPSEEK_API_KEY:
            logging.error("DEEPSEEK_API_KEY is not set in .env file. Please set it to use DeepSeek LLM.")
            raise ValueError("DEEPSEEK_API_KEY is not set.")

        self.llm = ChatDeepSeek(
            model=Config.DEEPSEEK_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            deepseek_api_key=Config.DEEPSEEK_API_KEY,
            # Add other DeepSeek specific parameters as needed from the docs
        )
        logging.info(f"DeepSeek LLM initialized with model: {Config.DEEPSEEK_MODEL_NAME}")

    def get_llm(self):
        """ Returns the initialized DeepSeek LLM instance. """
        return self.llm

    def invoke(self, prompt: str):
        """ Invokes the LLM with a given prompt and returns the response. """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logging.error(f"Error invoking DeepSeek LLM: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Make sure to set DEEPSEEK_API_KEY in your .env file before running this example
    try:
        deepseek_llm = DeepSeekLLM()
        llm_instance = deepseek_llm.get_llm()
        print(f"\nDeepSeek LLM instance: {llm_instance}")

        test_prompt = "你好，请用中文介绍一下你自己。"
        print(f"\nPrompt: {test_prompt}")
        response = deepseek_llm.invoke(test_prompt)
        print(f"Response: {response}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
