import os
import pytest
import json
from dotenv import load_dotenv
from app.utils.openai_client import OpenAIClient

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def client():
    """Create an OpenAI client for testing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAIClient(api_key=api_key)

def test_standard_chat(client):
    """Test a standard chat completion query to OpenAI."""
    print("\n=== Testing Standard Chat Completion ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = client.chat(messages=messages, max_tokens=50)
    print(f"Response: {response['content']}")
    assert response["success"] is True
    assert response["content"] is not None
    assert isinstance(response["content"], str)
    assert len(response["content"]) > 0
    assert "Paris" in response["content"]

def test_single_embedding(client):
    """Test creating a single embedding."""
    print("\n=== Testing Single Embedding ===")
    text = "OpenAI provides powerful AI models."
    response = client.embedding(input_text=text)
    print(f"Embedding: {response['embedding'][:5]}... (length: {len(response['embedding']) if response['embedding'] else 0})")
    assert response["success"] is True
    assert response["embedding"] is not None
    assert isinstance(response["embedding"], list)
    assert len(response["embedding"]) > 0

def test_batch_embeddings(client):
    """Test creating embeddings for a batch of texts."""
    print("\n=== Testing Batch Embeddings ===")
    texts = [
        "OpenAI provides powerful AI models.",
        "Paris is the capital of France.",
        "Python is a popular programming language."
    ]
    response = client.embeddings_batch(texts=texts)
    print(f"Embeddings count: {len(response['embeddings']) if response['embeddings'] else 0}")
    assert response["success"] is True
    assert response["embeddings"] is not None
    assert isinstance(response["embeddings"], list)
    assert len(response["embeddings"]) == len(texts)
    for emb in response["embeddings"]:
        assert isinstance(emb, list)
        assert len(emb) > 0 