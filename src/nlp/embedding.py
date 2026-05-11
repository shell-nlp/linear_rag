from langchain_openai import OpenAIEmbeddings
from src.core.config import LLM_BASE_URL


class LocalOpenAIEmbeddingModel:
    """
    一个用于调用本地OpenAI兼容Embedding API的客户端类。
    """

    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key="123",
            openai_api_base=LLM_BASE_URL,
        )

    def encode(self, sentences, batch_size=32, **kwargs) -> list[list[float]]:
        """
        生成句子嵌入。
        **kwargs 用于接收并忽略来自上层调用的、本方法不支持的额外参数。
        """
        result = self.embedding_model.embed_documents(sentences)
        return result
