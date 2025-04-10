"""Smoke tests for the Anthropic client with real API calls."""

import os
import pytest
import json
from app.utils.anthropic_client import AnthropicClient


@pytest.fixture
def client():
    """Create an Anthropic client for testing."""
    # Skip tests if API key is not set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    return AnthropicClient(api_key=api_key)


def test_standard_query(client):
    """Test a standard query to Claude."""
    print("\n=== Testing Standard Query ===")
    response = client.query(
        prompt="What is the capital of France?",
        max_tokens=100
    )
    
    print(f"Response: {response}")
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Paris" in response


def test_extended_thinking(client):
    """Test a query with extended thinking enabled."""
    print("\n=== Testing Extended Thinking ===")
    result = client.query_with_extended_thinking(
        prompt="What are three potential solutions to climate change?",
        thinking_budget=2000,
        max_tokens=3000
    )
    
    print(f"Thinking: {result['thinking'][:300]}...\n")
    print(f"Response: {result['response'][:300]}...\n")
    
    assert result is not None
    assert isinstance(result, dict)
    assert "thinking" in result
    assert "response" in result
    assert len(result["thinking"]) > 0
    assert len(result["response"]) > 0


def test_tool_usage(client):
    """Test tool usage with Claude."""
    print("\n=== Testing Tool Usage ===")
    # Define a simple calculator tool
    calculator_tool = {
        "name": "calculator",
        "description": "A calculator tool that can perform basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
    
    # Make the request with tool
    response = client.query_with_tools(
        prompt="What is 25 multiplied by 4?",
        tools=[calculator_tool]
    )
    
    # Check if Claude tried to use the calculator tool
    tool_use_found = False
    for content_block in response.content:
        if hasattr(content_block, "type") and content_block.type == "tool_use":
            tool_use_found = True
            print(f"Tool name: {content_block.name}")
            print(f"Tool input: {json.dumps(content_block.input, indent=2)}")
            break
    
    assert tool_use_found, "Claude did not attempt to use the calculator tool"


def test_complete_tool_conversation(client):
    """Test a complete tool conversation with request and response."""
    print("\n=== Testing Complete Tool Conversation ===")
    calculator_tool = {
        "name": "calculator",
        "description": "A calculator tool that can perform basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
    
    # First message with tool
    first_response = client.query_with_tools(
        prompt="What is 17 multiplied by 13?",
        tools=[calculator_tool]
    )
    
    # Find the tool use request
    tool_use_block = None
    for block in first_response.content:
        if hasattr(block, "type") and block.type == "tool_use":
            tool_use_block = block
            print(f"Tool name: {block.name}")
            print(f"Tool input: {json.dumps(block.input, indent=2)}")
            break
    
    assert tool_use_block is not None, "Claude did not attempt to use the calculator tool"
    
    # Calculate the result (in a real app, you'd call the actual tool function)
    expression = tool_use_block.input.get("expression")
    # Don't use eval in real code - this is just for testing
    result = str(eval(expression)) if expression else "Error: No expression provided"
    print(f"Calculator result: {result}")
    
    # Continue the conversation with the tool result
    tool_result = [{
        "type": "tool_result",
        "tool_use_id": tool_use_block.id,
        "content": result
    }]
    
    conversation_history = [
        {"role": "user", "content": "What is 17 multiplied by 13?"},
        {"role": "assistant", "content": [tool_use_block]}
    ]
    
    # Get the follow-up response
    follow_up = client.continue_tool_conversation(
        conversation_history=conversation_history,
        tool_results=tool_result
    )
    
    # Look for text blocks in the response
    text_found = False
    for block in follow_up.content:
        if hasattr(block, "type") and block.type == "text":
            text_found = True
            print(f"Final response: {block.text}")
            assert len(block.text) > 0
            assert "221" in block.text  # 17 * 13 = 221
            break
    
    assert text_found, "Claude did not provide a text response with the calculation result"
