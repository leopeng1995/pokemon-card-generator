import os

from retry import retry
from dotenv import load_dotenv
from functools import cached_property
from zhipuai import ZhipuAI


class ZhipuAIClient:

    SINGLETON_CLIENT = None

    def __init__(self):
        if not self.enabled:
            raise ValueError("ZhipuAI API key not found.")
        api_key = os.getenv("ZHIPUAI_API_KEY")
        self.client = ZhipuAI(api_key=api_key)

    @cached_property
    def enabled(self):
        print("Checking for ZhipuAI API key...")
        load_dotenv()
        if os.getenv("ZHIPUAI_API_KEY") is None or os.getenv("ZHIPUAI_API_KEY") == "":
            # Print warning message in red.
            print(
                "\033[91m"
                + "WARNING: ZhipuAI API key not found. ZhipuAI will not be used."
                + "\033[0m"
            )
            return False
        else:
            return True

    @retry(tries=3, delay=3.0)
    def get_completion(self, prompt: str, max_tokens: int = 128):
        load_dotenv()

        messages = [
            { "role": "system", "content": "你是一个英语老师，使用英语回答问题，不要出现中文" },
            { "role": "user", "content": prompt },
        ]

        response = self.client.chat.completions.create(
            model="glm-3-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.9,
        )
        return response


def gpt_client(provider: str = "zhipuai"):
    if ZhipuAIClient.SINGLETON_CLIENT is None:
        ZhipuAIClient.SINGLETON_CLIENT = ZhipuAIClient()
    return ZhipuAIClient.SINGLETON_CLIENT
