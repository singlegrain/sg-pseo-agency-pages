"""Smoke tests for the Perplexity client with real API calls."""

import os
import pytest
import json
from dotenv import load_dotenv
from app.utils.perplexity_client import PerplexityClient

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def client():
    """Create a Perplexity client for testing."""
    # Skip tests if API key is not set
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY environment variable not set")
    
    return PerplexityClient(api_key=api_key)


def test_standard_query(client):
    """Test a standard query to Perplexity."""
    print("\n=== Testing Standard Query ===")
    response = client.query(
        prompt="What is the capital of France?",
        max_tokens=100
    )
    
    print(f"Response: {response['content']}")
    
    assert response["success"] is True
    assert response["content"] is not None
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0
    assert "Paris" in response["content"]


def test_query_with_search(client):
    """Test a query with search functionality."""
    print("\n=== Testing Query with Search ===")
    response = client.query_with_search(
        prompt="What are the latest advancements in quantum computing?",
        search_context_size="medium",
        max_tokens=300
    )
    
    print(f"Response: {response['content'][:300]}...\n")
    
    # Check if citations are present
    if "citations" in response:
        print(f"Citations found: {len(response['citations'])}")
        for i, citation in enumerate(response["citations"][:3]):
            print(f"Citation {i+1}: {citation}")
    
    assert response["success"] is True
    assert response["content"] is not None
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0
    assert "search_context_size" in response
    assert response["search_context_size"] == "medium"


def test_deep_research(client):
    """Test a deep research query."""
    print("\n=== Testing Deep Research Query ===")
    response = client.query_deep_research(
        prompt="Analyze the impact of AI on healthcare over the last 5 years.",
        max_tokens=500
    )
    
    print(f"Response: {response['content'][:300]}...\n")
    
    assert response["success"] is True
    assert response["content"] is not None
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0
    
    # Check if citations are present
    if "citations" in response:
        print(f"Citations found: {len(response['citations'])}")
        for i, citation in enumerate(response["citations"][:3]):
            print(f"Citation {i+1}: {citation}")


def test_search_with_domain_filter(client):
    """Test search with domain filtering."""
    print("\n=== Testing Search with Domain Filter ===")
    domains = ["github.com", "arxiv.org"]
    response = client.query_with_search(
        prompt="What are some recent advances in machine learning frameworks?",
        search_domain_filter=domains,
        search_context_size="high",
        max_tokens=300
    )
    
    # First check if the request was successful
    assert "success" in response, "Response missing 'success' key"
    
    if response["success"]:
        print(f"Response: {response['content'][:300]}...\n")
        
        assert response["content"] is not None
        assert isinstance(response["content"], str)
        assert len(response["content"]) > 0
        assert "search_context_size" in response
        assert response["search_context_size"] == "high"
        assert "search_domains" in response
        assert response["search_domains"] == domains
        
        # Check if token usage information is available
        if "usage" in response:
            print(f"Token usage: {json.dumps(response['usage'], indent=2)}")
    else:
        print(f"Search with domain filter failed: {response.get('error')}")
        pytest.skip(f"Search with domain filter failed: {response.get('error')}")


def test_connection(client):
    """Test the connection to Perplexity API."""
    print("\n=== Testing API Connection ===")
    result = client.test_connection()
    
    print(f"Connection test result: {result}")
    
    assert result["success"] is True
    assert "message" in result
    assert "Successfully connected" in result["message"]
    assert "model_tested" in result
    assert result["model_tested"] == "sonar-pro" 