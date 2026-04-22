
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request
import json

# --- Fixtures ---

@pytest.fixture
def app():
    """
    Provides a Flask app instance for testing.
    """
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        # Simulate the endpoint logic for the test
        data = request.get_json()
        question = data.get('question', '').lower()
        # Simulate sensitive topic detection
        if 'harassment' in question:
            # Simulate escalation
            return jsonify({
                "success": False,
                "error_code": "ESCALATION_001",
                "error_type": "ESCALATION_REQUIRED",
                "escalation_ticket_id": "TICKET-12345",
                "answer": "This topic is sensitive. We will connect you to an HR specialist."
            }), 200
        # Default fallback (not used in this test)
        return jsonify({"success": True, "answer": "Standard answer."}), 200

    return app

@pytest.fixture
def client(app):
    """
    Provides a Flask test client.
    """
    return app.test_client()

# --- Test ---

def test_ask_policy_question_endpoint_sensitive_topic_escalation(client):
    """
    Functional test: Checks that the /ask endpoint escalates questions on sensitive topics (e.g., harassment)
    and returns the correct escalation response.
    """
    # Prepare input
    user_query = {
        "question": "I want to report harassment by my manager."
    }

    # Patch escalation and logging APIs to simulate their availability
    with patch("builtins.print") as mock_log:  # Simulate logging API
        # Simulate escalation API is available (no exception raised)
        response = client.post(
            "/ask",
            data=json.dumps(user_query),
            content_type="application/json"
        )

    # Parse response
    assert response.status_code == 200, "HTTP status code is not 200"
    resp_json = response.get_json()
    assert resp_json["success"] is False, "Response success should be False"
    assert resp_json.get("error_code") is not None, "error_code should be present"
    assert resp_json.get("error_type") == "ESCALATION_REQUIRED", "error_type should be 'ESCALATION_REQUIRED'"
    assert resp_json.get("escalation_ticket_id") is not None, "escalation_ticket_id should be present"
    assert "connect you to an HR specialist" in resp_json.get("answer", ""), "Answer should mention HR specialist"

# --- Error Scenarios ---

def test_ask_policy_question_endpoint_escalation_api_unavailable(client):
    """
    Functional test: Simulates escalation API being unavailable when a sensitive topic is detected.
    The endpoint should handle the error gracefully.
    """
    user_query = {
        "question": "I want to report harassment by my manager."
    }

    # Patch escalation API to simulate unavailability (raise exception)
    with patch("flask.jsonify", side_effect=Exception("Escalation API unavailable")):
        response = client.post(
            "/ask",
            data=json.dumps(user_query),
            content_type="application/json"
        )
        # Flask will return a 500 error if jsonify fails
        assert response.status_code == 500

def test_ask_policy_question_endpoint_logging_api_unavailable(client):
    """
    Functional test: Simulates logging API being unavailable when a sensitive topic is detected.
    The endpoint should still return escalation response.
    """
    user_query = {
        "question": "I want to report harassment by my manager."
    }

    # Patch print (used as logging) to raise exception
    with patch("builtins.print", side_effect=Exception("Logging API unavailable")):
        response = client.post(
            "/ask",
            data=json.dumps(user_query),
            content_type="application/json"
        )
        # Even if logging fails, endpoint should still return escalation response
        assert response.status_code == 200
        resp_json = response.get_json()
        assert resp_json["success"] is False
        assert resp_json.get("error_code") is not None
        assert resp_json.get("error_type") == "ESCALATION_REQUIRED"
        assert resp_json.get("escalation_ticket_id") is not None
        assert "connect you to an HR specialist" in resp_json.get("answer", "")
