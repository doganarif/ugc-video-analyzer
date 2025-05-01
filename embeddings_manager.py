import os
from typing import List
import asyncio
from openai import OpenAI, AzureOpenAI
from config import get_openai_api_key

class EmbeddingsManager:
    """Manager for handling text embeddings generation"""

    def __init__(self, api_key=None, model="text-embedding-3-large", use_azure=True, azure_endpoint=None, azure_api_version=None, azure_deployment="text-embedding-3-large"):
        """Initialize the embeddings manager with an OpenAI API key"""
        self.api_key = api_key or get_openai_api_key()
        self.model = model
        self.use_azure = use_azure
        self.azure_deployment = azure_deployment

        if use_azure:
            if not azure_endpoint:
                azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_api_version:
                azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

            if not azure_endpoint or not azure_api_version:
                raise ValueError("Azure OpenAI endpoint and API version are required for Azure mode")

            self.client = AzureOpenAI(
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                api_key=self.api_key
            )
        else:
            self.client = OpenAI(api_key=self.api_key)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a single text string"""
        # When using Azure, we use the deployment name instead of model name
        if self.use_azure:
            response = self.client.embeddings.create(
                model=self.azure_deployment,
                input=text,
                dimensions=1536
            )
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=1536
            )
        return response.data[0].embedding

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embeddings for a single text string asynchronously"""
        # Use asyncio to run the synchronous method in a thread pool
        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel"""
        tasks = [self.generate_embedding_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
