"""Basic tests for the pSEO agency pages generator."""

import pytest
from app.main import main


def test_main_runs_without_error():
    """Test that the main function runs without raising exceptions."""
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"Main function raised exception: {e}")


# Add more specific tests as the application grows
