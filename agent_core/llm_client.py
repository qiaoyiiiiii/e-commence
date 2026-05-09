import logging

from langchain_openai import ChatOpenAI

from config import Config


class DeepSeekLLM:
    def __init__(self):
        if not Config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content
