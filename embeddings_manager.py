import os
import logging
from typing import List
import asyncio
from openai import OpenAI, AzureOpenAI
from config import get_openai_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manager for handling text embeddings generation"""

    def __init__(self, api_key=None, model="text-embedding-3-large", use_azure=True, azure_endpoint=None, azure_api_version=None, azure_deployment="text-embedding-3-large"):
        """
        Initialize the embeddings manager with OpenAI configuration
        
        Args:
            api_key (str, optional): API key for OpenAI
            model (str, optional): Model name for embeddings
            use_azure (bool, optional): Whether to use Azure OpenAI
            azure_endpoint (str, optional): Azure endpoint URL
            azure_api_version (str, optional): Azure API version
            azure_deployment (str, optional): Azure deployment name
        """
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
            logger.info(f"Using Azure OpenAI for embeddings with model {azure_deployment}")
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Using OpenAI for embeddings with model {model}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string
        
        Args:
            text (str): Text to generate embeddings for
            
        Returns:
            List[float]: Vector of embeddings
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
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
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string asynchronously
        
        Args:
            text (str): Text to generate embeddings for
            
        Returns:
            List[float]: Vector of embeddings
        """
        logger.debug(f"Generating embedding asynchronously for text of length {len(text)}")
        # Use asyncio to run the synchronous method in a thread pool
        return await asyncio.to_thread(self.generate_embedding, text)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        logger.info(f"Generating embeddings for batch of {len(texts)} texts")
        tasks = [self.generate_embedding_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
