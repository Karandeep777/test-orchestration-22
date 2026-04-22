
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Assume the FastAPI app is defined in app.main as 'app'
# and the endpoint is /ask
from app.main import app

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def valid_user_query():
    """Fixture to provide a valid user query payload."""
    return {
        "user_id": "user123",
        "query": "What is the vacation policy for new employees?"
    }

def mock_openai_success(*args, **kwargs):
    """Mocked OpenAI API response for a successful LLM call."""
    class MockResponse:
        def __init__(self):
            self.choices = [{"text": "The vacation policy for new employees is ..."}]
    return MockResponse()

def mock_azure_search_success(*args, **kwargs):
    """Mocked Azure Search API response."""
    return {
        "cited_source": "HR Handbook",
        "cited_section": "Vacation Policy",
        "cited_last_updated": "2023-01-01"
    }

def mock_redis_success(*args, **kwargs):
    """Mocked Redis get/set always succeeds."""
    return None

class MockRedis:
    def get(self, *args, **kwargs):
        return None
    def set(self, *args, **kwargs):
        return True

def build_formatted_response():
    return {
        "success": True,
        "answer": "The vacation policy for new employees is ...",
        "cited_source": "HR Handbook",
        "cited_section": "Vacation Policy",
        "cited_last_updated": "2023-01-01",
        "error_code": None
    }

def patch_all_success():
    """
    Helper context manager to patch all external dependencies for a successful /ask call.
    """
    # Patch OpenAI LLM call, Azure Search, and Redis
    return (
        patch("app.services.llm_client.LLMClient.ask", return_value="The vacation policy for new employees is ..."),
        patch("app.services.search_client.SearchClient.search", return_value={
            "cited_source": "HR Handbook",
            "cited_section": "Vacation Policy",
            "cited_last_updated": "2023-01-01"
        }),
        patch("app.services.redis_client.get_redis", return_value=MockRedis())
    )

def test_ask_policy_question_endpoint_success(client, valid_user_query):
    """
    Functional test: Validates that the /ask endpoint returns a successful response
    with a valid answer when provided with a well-formed user query.
    """
    with patch_all_success() as (mock_llm, mock_search, mock_redis):
        response = client.post("/ask", json=valid_user_query)
        assert response.status_code == 200, "Expected HTTP 200 OK"
        data = response.json()
        assert data["success"] is True, "Expected success=True"
        assert isinstance(data["answer"], str) and data["answer"], "Answer should be a non-empty string"
        # cited_source, cited_section, cited_last_updated can be present or None
        assert "cited_source" in data
        assert "cited_section" in data
        assert "cited_last_updated" in data
        assert data["error_code"] is None, "error_code should be None on success"
