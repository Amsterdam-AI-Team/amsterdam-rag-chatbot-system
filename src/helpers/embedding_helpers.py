from openai import OpenAI
from typing import List
import config as cf
import os

class OpenAIEmbeddingFunction:
    def __init__(self, model: str = "text-embedding-3-large"):
        client = OpenAI(
            api_key=cf.API_KEYS["openai"],
        )
        self.model = model
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model=self.model)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding