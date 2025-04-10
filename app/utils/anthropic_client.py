"""Anthropic Client for interacting with Claude models."""

import os
import anthropic
from typing import Dict, List, Optional, Union, Any

class AnthropicClient:
    """Client for interacting with Anthropic's Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If None, it will be loaded from the ANTHROPIC_API_KEY env variable.
            model: The Claude model to use. Defaults to "claude-3-7-sonnet-20250219".
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
    
    def query(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Make a standard query to Claude.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system prompt to control Claude's behavior
            max_tokens: Maximum number of tokens in the response
            temperature: Controls randomness (0-1)
            
        Returns:
            The text response from Claude
        """
        # Create message parameters
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
        
        # Make the API call
        response = self.client.messages.create(**params)
        
        return response.content[0].text
    
    def query_with_extended_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 16000,
        thinking_budget: int = 8000,
        temperature: Optional[float] = None  # No longer used, always 1.0
    ) -> Dict[str, Any]:
        """Make a query to Claude with extended thinking enabled.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system prompt to control Claude's behavior
            max_tokens: Maximum number of tokens in the response (including thinking)
            thinking_budget: Maximum tokens to allocate for thinking
            temperature: Not used - extended thinking requires temperature=1.0
            
        Returns:
            Dict containing both thinking content and final response
        """
        # Create message parameters
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 1.0,  # Extended thinking requires temperature=1.0
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
        
        # Make the API call
        response = self.client.messages.create(**params)
        
        # Extract thinking and final response
        thinking_content = ""
        final_response = ""
        
        for content_block in response.content:
            if content_block.type == "thinking":
                thinking_content = content_block.thinking
            elif content_block.type == "text":
                final_response = content_block.text
                
        return {
            "thinking": thinking_content,
            "response": final_response
        }
    
    def query_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        tool_choice: Optional[Dict[str, Any]] = None,
        token_efficient: bool = False
    ) -> anthropic.types.Message:
        """Make a query to Claude with tool usage.
        
        Args:
            prompt: The user's prompt/question
            tools: List of tool definitions with name, description, and input_schema
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Controls randomness (0-1)
            tool_choice: Optional tool choice configuration
            token_efficient: Whether to use token-efficient tool mode (beta)
            
        Returns:
            The full message response from Claude
        """
        # Create message parameters
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
            
        # Add tool_choice if provided
        if tool_choice:
            params["tool_choice"] = tool_choice
        else:
            params["tool_choice"] = {"type": "auto"}
            
        # Add beta header for token-efficient tools if requested
        if token_efficient and self.model == "claude-3-7-sonnet-20250219":
            # Note: This needs to be handled differently - the header needs to be passed
            # directly to the API call, which isn't supported in this simple wrapper.
            # In a real implementation, we'd need to use the underlying API client's
            # request methods to set this header.
            pass
            
        # Make the API call
        response = self.client.messages.create(**params)
        
        return response
    
    def continue_tool_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> anthropic.types.Message:
        """Continue a conversation after receiving tool results.
        
        Args:
            conversation_history: Previous messages in the conversation
            tool_results: Results from tools used in the previous response
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            The full message response from Claude
        """
        # Prepare the new user message with tool results
        user_message = {
            "role": "user",
            "content": tool_results
        }
        
        # Add the new message to the conversation history
        messages = conversation_history + [user_message]
        
        # Create message parameters
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = system_prompt
        
        # Send the request
        response = self.client.messages.create(**params)
        
        return response
