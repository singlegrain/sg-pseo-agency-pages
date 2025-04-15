import os
from typing import List, Optional, Dict, Any
from openai import OpenAI

class OpenAIClient:
    """Client for interacting with OpenAI models (chat completions and embeddings)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from the OPENAI_API_KEY env variable.
            model: The default OpenAI chat model to use.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenAI.
        
        Args:
            messages: List of message dicts (role: 'system'|'user'|'assistant', content: str)
            model: The model to use (defaults to the client's model if not specified)
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens in the response (None for default)
            timeout: Maximum time to wait for a response in seconds
        
        Returns:
            Dict containing the response and metadata
        """
        model = model or self.model
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout
        }
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        try:
            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
            result = {
                "success": True,
                "content": content,
                "model": model,
                "usage": getattr(response, 'usage', None),
                "error": None
            }
            return result
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "model": model,
                "usage": None,
                "error": str(e)
            }

    def embedding(
        self,
        input_text: str,
        model: str = "text-embedding-ada-002",
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Create an embedding for a single text using OpenAI.
        
        Args:
            input_text: The text to embed
            model: The embedding model to use
            timeout: Timeout for the API call
        
        Returns:
            Dict containing the embedding and metadata
        """
        try:
            response = self.client.embeddings.create(
                input=input_text,
                model=model,
                timeout=timeout
            )
            embedding = response.data[0].embedding
            return {
                "success": True,
                "embedding": embedding,
                "model": model,
                "usage": getattr(response, 'usage', None),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "embedding": None,
                "model": model,
                "usage": None,
                "error": str(e)
            }

    def embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        timeout: float = 60.0,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Create embeddings for a batch of texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            model: The embedding model to use
            timeout: Timeout for the API call
            batch_size: Number of texts per batch
        
        Returns:
            Dict containing the list of embeddings and metadata
        """
        all_embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=model,
                    timeout=timeout
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            return {
                "success": True,
                "embeddings": all_embeddings,
                "model": model,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "embeddings": None,
                "model": model,
                "error": str(e)
            } 