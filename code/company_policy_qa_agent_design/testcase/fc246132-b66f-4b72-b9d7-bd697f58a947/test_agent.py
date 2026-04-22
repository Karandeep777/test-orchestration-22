
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request

@pytest.fixture
def app():
    """
    Fixture to create a Flask app instance with the /admin/update_kb endpoint.
    The endpoint simulates a knowledge base update and returns error responses on failure.
    """
    app = Flask(__name__)

    @app.route('/admin/update_kb', methods=['POST'])
    def update_kb():
        try:
            # Simulate parsing JSON and updating KB
            data = request.get_json(force=True)
            # Simulate a failure in the update process
            raise RuntimeError("Knowledge base update failed")
        except Exception as e:
            return jsonify({
                "success": False,
                "error_code": "KB_UPDATE_FAILED",
                "error_type": "UpdateError",
                "fixing_tips": "Check the document format and try again."
            }), 500

    return app

@pytest.fixture
def client(app):
    """
    Fixture to provide a Flask test client for the app.
    """
    return app.test_client()

def test_update_knowledge_base_endpoint_failure(client):
    """
    Ensures that the /admin/update_kb endpoint returns an appropriate error response
    when the knowledge base update fails due to malformed JSON or internal error.
    """
    # Simulate malformed JSON (invalid payload)
    response = client.post(
        '/admin/update_kb',
        data='{"invalid_json": ',  # Malformed JSON
        content_type='application/json'
    )
    assert response.status_code == 500, "Expected HTTP 500 for malformed JSON"
    resp_json = response.get_json()
    assert resp_json["success"] is False, "Expected success to be False"
    assert resp_json["error_code"] == "KB_UPDATE_FAILED", "Expected error_code to be 'KB_UPDATE_FAILED'"
    assert resp_json["error_type"] == "UpdateError", "Expected error_type to be 'UpdateError'"
    assert resp_json.get("fixing_tips") is not None, "Expected fixing_tips to be present"

    # Simulate valid JSON but HR Admin API returns error (simulate update failure)
    with patch("flask.request.get_json", return_value={"doc": "valid_but_triggers_error"}):
        response2 = client.post(
            '/admin/update_kb',
            json={"doc": "valid_but_triggers_error"}
        )
        assert response2.status_code == 500, "Expected HTTP 500 for update failure"
        resp_json2 = response2.get_json()
        assert resp_json2["success"] is False, "Expected success to be False"
        assert resp_json2["error_code"] == "KB_UPDATE_FAILED", "Expected error_code to be 'KB_UPDATE_FAILED'"
        assert resp_json2["error_type"] == "UpdateError", "Expected error_type to be 'UpdateError'"
        assert resp_json2.get("fixing_tips") is not None, "Expected fixing_tips to be present"
