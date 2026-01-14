import json
import requests
import numpy as np

class LocalOpenAIEmbeddingModel:
    """
    一个用于调用本地OpenAI兼容Embedding API的客户端类。
    """
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}


    def encode(self, sentences, batch_size=32, **kwargs):
        """
        生成句子嵌入。
        **kwargs 用于接收并忽略来自上层调用的、本方法不支持的额外参数。
        """
        all_embeddings = []
        # 打印信息以模拟进度条
        print(f"正在编码 {len(sentences)} 个文本片段，批处理大小: {batch_size}...")
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            payload = {"input": batch, "model": self.model_name}
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
                response.raise_for_status()
                embeddings = [item['embedding'] for item in response.json()['data']]
                all_embeddings.extend(embeddings)
            except requests.exceptions.RequestException as e:
                print(f"调用Embedding API时出错: {e}")
                # 打印更详细的错误信息
                if response is not None:
                    print(f"API响应状态码: {response.status_code}")
                    print(f"API响应内容: {response.text}")
                return np.array([])
        print("编码完成。")
        result = np.array(all_embeddings)
        if len(result) > 0:
            print(f"DEBUG: API返回的嵌入维度形状为: {result.shape}")
        return result