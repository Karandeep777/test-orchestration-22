
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request
import json

# Assuming the app and endpoints are defined in app.py
# and the Flask app is named 'app'
# If using FastAPI or another framework, adjust accordingly.

@pytest.fixture
def client():
    """
    Pytest fixture to provide a Flask test client.
    """
    from app import app  # Import your Flask app here
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def user_query_payload():
    """
    Returns a valid UserQuery JSON payload with an unrelated question.
    """
    return {
        "query": "What is the weather today?",
        "user_id": "user123",
        "context": {}
    }

def mock_azure_search_no_results(*args, **kwargs):
    """
    Mocks Azure Search API to return no results.
    """
    return []

def mock_logging_api_unavailable(*args, **kwargs):
    """
    Mocks logging API to raise a connection error.
    """
    raise ConnectionError("Logging API unavailable")

def build_expected_response():
    """
    Helper to build the expected error response.
    """
    return {
        "success": False,
        "error_code": "NO_ANSWER_FOUND",
        "error_message": "No relevant policy information found",
        "fixing_tips": "Try rephrasing your question or check the policy topics.",
        "data": None
    }

class TestAskPolicyQuestionEndpoint:
    def test_ask_policy_question_endpoint_no_relevant_policy_found(
        self, client, user_query_payload
    ):
        """
        Functional test:
        Checks that the /ask endpoint returns a proper error response when no relevant policy information is found for the query.
        Simulates Azure Search API returning no results and logging API being unavailable.
        """
        # Patch Azure Search API call to return empty result
        # Patch logging API to simulate unavailability (should not break the endpoint)
        # Patch any external HTTP calls to prevent real network connections

        with patch("app.azure_search_api.search", side_effect=mock_azure_search_no_results), \
             patch("app.logging_api.log", side_effect=mock_logging_api_unavailable), \
             patch("requests.post") as mock_requests_post:
            # Patch requests.post to prevent any real HTTP calls (e.g., if used for logging)
            mock_requests_post.return_value = MagicMock(status_code=200, json=lambda: {"logged": True})

            response = client.post(
                "/ask",
                data=json.dumps(user_query_payload),
                content_type="application/json"
            )

            assert response.status_code == 200, "HTTP status code is not 200"

            resp_json = response.get_json()
            assert resp_json is not None, "Response is not valid JSON"

            assert resp_json.get("success") is False, "Response 'success' should be False"
            assert resp_json.get("error_code") == "NO_ANSWER_FOUND", "Error code should be 'NO_ANSWER_FOUND'"
            assert "No relevant policy information found" in resp_json.get("error_message", ""), "Error message should mention no relevant policy"
            assert resp_json.get("fixing_tips") is not None, "Fixing tips should be present"

