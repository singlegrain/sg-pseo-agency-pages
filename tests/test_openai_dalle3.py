import os
import pytest
import json
from dotenv import load_dotenv
from app.utils.openai_client import OpenAIClient

# Load environment variables from .env file
load_dotenv()

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dalle3_test_output.json")

@pytest.fixture
def client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAIClient(api_key=api_key)

def test_dalle3_generate_image(client):
    """Smoke test for DALL·E 3 image generation with a minimal prompt."""
    print("\n=== Testing DALL·E 3 Image Generation ===")
    prompt = "marketing AI evolution timeline, icons, blue and orange"
    response = client.generate_image(prompt=prompt, n=1, size="1024x1024", quality="standard", response_format="url")
    print(f"Response: {json.dumps(response, indent=2)}")

    assert response["success"] is True
    assert response["images"] is not None
    assert isinstance(response["images"], list)
    assert len(response["images"]) == 1
    assert response["images"][0].startswith("http")
    print(f"Generated image link: {response['images'][0]}") 