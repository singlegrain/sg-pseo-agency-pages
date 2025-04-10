"""Perplexity Client for interacting with Perplexity AI models."""

import os
import time
import re
from typing import Dict, List, Optional, Union, Any

from openai import OpenAI


class PerplexityClient:
    """Client for interacting with Perplexity AI models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "sonar-pro", base_url: str = "https://api.perplexity.ai"):
        """Initialize the Perplexity client.
        
        Args:
            api_key: Perplexity API key. If None, it will be loaded from the PERPLEXITY_API_KEY env variable.
            model: The default Perplexity model to use. Defaults to "sonar-pro".
            base_url: The base URL for the Perplexity API.
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not provided and not found in environment variables")
        
        self.base_url = base_url
        self.model = model
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def query(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
        retries: int = 3,
        retry_delay: int = 2,
        remove_thinking: bool = True
    ) -> Dict[str, Any]:
        """Send a query to Perplexity AI.
        
        Args:
            prompt: The user's prompt/question
            model: The model to use (defaults to the client's model if not specified)
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens in the response (None for default)
            temperature: Controls randomness (0-1)
            timeout: Maximum time to wait for a response in seconds
            retries: Number of retry attempts if the request fails
            retry_delay: Delay between retries in seconds
            remove_thinking: Whether to remove thinking process from deep research models
            
        Returns:
            Dict containing the query results and metadata
        """
        # Use the default model if none is specified
        model = model or self.model
        
        # Validate model selection
        valid_models = ["sonar-pro", "sonar-deep-research", "sonar-reasoning", "sonar-reasoning-pro", "sonar", "r1-1776"]
        if model not in valid_models:
            raise ValueError(f"Invalid model specified. Must be one of: {valid_models}")
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout
        }
        
        # Only add max_tokens if it's specified
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        # Try the request with retries
        for attempt in range(retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**api_params)
                elapsed_time = time.time() - start_time
                
                # Get content from the response
                content = response.choices[0].message.content
                
                # Process response to remove thinking process if requested for deep research model
                if remove_thinking and "deep-research" in model and "<think>" in content:
                    content = self._extract_final_answer(content)
                
                # Get citations from model_extra if available
                citations = None
                if hasattr(response, 'model_extra') and response.model_extra:
                    citations = response.model_extra.get('citations')
                
                # For backward compatibility, also check direct attribute
                if citations is None and hasattr(response, 'citations'):
                    citations = getattr(response, 'citations')
                
                result = {
                    "success": True,
                    "content": content,
                    "model": model,
                    "elapsed_time": elapsed_time,
                    "error": None
                }
                
                # Add citations to the result if they exist
                if citations is not None:
                    result["citations"] = citations
                
                # Add token usage information if available
                if hasattr(response, 'usage'):
                    result["usage"] = {
                        "total_tokens": getattr(response.usage, 'total_tokens', None),
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                        "citation_tokens": getattr(response.usage, 'citation_tokens', None),
                        "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', None),
                        "num_search_queries": getattr(response.usage, 'num_search_queries', None)
                    }
                
                return result
                
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    error_msg = f"Perplexity query error after {retries} attempts: {str(e)}"
                    print(error_msg)
                    return {
                        "success": False,
                        "content": None,
                        "model": model,
                        "elapsed_time": time.time() - start_time if 'start_time' in locals() else None,
                        "error": error_msg
                    }
    
    def _extract_final_answer(self, content: str) -> str:
        """Extract the final answer from a deep research model response by removing the thinking process.
        
        Args:
            content: The raw model response with thinking process
            
        Returns:
            The processed content with thinking process removed
        """
        # Check if the content has both opening and closing think tags
        if "<think>" in content and "</think>" in content:
            # Check if there's content after the closing think tag
            parts = content.split("</think>", 1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()
            
            # If there's no content after the closing tag, we need to extract the conclusion
            # from within the thinking process
            thinking_content = re.search(r'<think>(.*)</think>', content, re.DOTALL)
            if thinking_content:
                thinking_text = thinking_content.group(1).strip()
                
                # Look for concluding paragraphs within the thinking
                conclusion_markers = [
                    "In conclusion", "To summarize", "Therefore", "Thus", "In summary",
                    "The answer is", "Overall", "Based on", "To conclude", "In short",
                    "In essence", "Final analysis"
                ]
                
                # First try to find any conclusion markers
                for marker in conclusion_markers:
                    idx = thinking_text.rfind(marker)
                    if idx != -1:
                        return thinking_text[idx:].strip()
                
                # If no conclusion markers found, return the last paragraph
                paragraphs = thinking_text.split("\n\n")
                if paragraphs:
                    return paragraphs[-1].strip()
        
        # If we couldn't extract a clean answer, at least remove the tags
        return content.replace("<think>", "").replace("</think>", "")
    
    def query_with_search(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        search_context_size: str = "medium",
        search_domain_filter: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Send a query to Perplexity AI with specific search options.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system prompt to guide the model
            search_context_size: Size of search context (high, medium, low)
            search_domain_filter: Optional list of domains to filter search results
            max_tokens: Maximum tokens in response
            temperature: Controls randomness (0-1)
            
        Returns:
            Dict containing the query results with search information
        """
        # Validate search context size
        valid_sizes = ["high", "medium", "low"]
        if search_context_size not in valid_sizes:
            raise ValueError(f"Invalid search context size. Must be one of: {valid_sizes}")
        
        # Create search options
        search_options = {
            "search_context_size": search_context_size
        }
        
        # Add domain filter to search options if provided
        if search_domain_filter:
            search_options["domain_filter"] = search_domain_filter
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare API call parameters
        api_params = {
            "model": "sonar-pro",  # This is specific for search capabilities
            "messages": messages,
            "temperature": temperature,
            "web_search_options": search_options
        }
            
        # Add max_tokens if specified
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(**api_params)
            elapsed_time = time.time() - start_time
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract citations if available
            citations = None
            if hasattr(response, 'model_extra') and response.model_extra:
                citations = response.model_extra.get('citations')
            
            if citations is None and hasattr(response, 'citations'):
                citations = getattr(response, 'citations')
            
            result = {
                "success": True,
                "content": content,
                "model": "sonar-pro",
                "elapsed_time": elapsed_time,
                "search_context_size": search_context_size,
                "error": None
            }
            
            # Add citations and search domains if available
            if citations:
                result["citations"] = citations
                
            if search_domain_filter:
                result["search_domains"] = search_domain_filter
            
            # Add token usage if available
            if hasattr(response, 'usage'):
                result["usage"] = {
                    "total_tokens": getattr(response.usage, 'total_tokens', None),
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                    "citation_tokens": getattr(response.usage, 'citation_tokens', None),
                    "num_search_queries": getattr(response.usage, 'num_search_queries', None)
                }
            
            return result
            
        except Exception as e:
            error_msg = f"Perplexity search query error: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "content": None,
                "model": "sonar-pro",
                "elapsed_time": time.time() - start_time if 'start_time' in locals() else None,
                "error": error_msg
            }
    
    def query_deep_research(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        remove_thinking: bool = True
    ) -> Dict[str, Any]:
        """Send a deep research query to Perplexity AI.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum tokens in response
            remove_thinking: Whether to remove thinking process from response
            
        Returns:
            Dict containing the deep research results
        """
        return self.query(
            prompt=prompt,
            model="sonar-deep-research",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=1.0,  # Deep research works best with temperature 1.0
            remove_thinking=remove_thinking
        )
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the Perplexity API.
        
        Returns:
            Dict with connection status and details
        """
        try:
            # Send a simple query to test the connection
            result = self.query(
                prompt="What is 1+1?",
                model="sonar-pro",
                max_tokens=10,  # Keep a small limit for the connection test
                temperature=0.0,
                timeout=30.0
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Successfully connected to Perplexity API",
                    "model_tested": "sonar-pro",
                    "api_base_url": self.base_url
                }
            else:
                return {
                    "success": False,
                    "message": f"Connection test failed: {result['error']}",
                    "error": result["error"]
                }
                
        except Exception as e:
            error_msg = f"Failed to connect to Perplexity API: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }
