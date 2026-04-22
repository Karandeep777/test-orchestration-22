
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Assuming the FastAPI app is defined in app.main as 'app'
# and StepResponse is defined in app.schemas
# Adjust import paths as needed for your project structure
try:
    from app.main import app
    from app.schemas import StepResponse
except ImportError:
    # Fallback for demonstration if actual modules are not present
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI()

    class StepResponse(BaseModel):
        step: str
        estimated_time: int
        next_action: str

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def valid_user_context():
    """Fixture for a valid UserContext payload."""
    return {
        "employee_id": "EMP12345",
        "role": "Software Engineer",
        "department": "Engineering",
        "operating_system": "Windows"
    }

def mock_step_response(*args, **kwargs):
    """Helper to return a mock StepResponse dict."""
    return {
        "step": "Install VPN client",
        "estimated_time": 10,
        "next_action": "Download and install the company VPN client from the provided link."
    }

def test_start_onboarding_functional_happy_path(client, valid_user_context):
    """
    Functional test: Validates that the /start endpoint successfully initiates an onboarding session
    and returns the first onboarding step.
    """
    # Patch any external dependencies (e.g., LLM, DB, etc.) used in the onboarding logic.
    # Here, we patch the function that generates the onboarding step.
    # Adjust the patch target to your actual implementation.
    with patch("app.routers.onboarding.generate_first_step", side_effect=mock_step_response):
        response = client.post("/start", json=valid_user_context)
        assert response.status_code == 200, "Expected HTTP 200 for valid onboarding start"
        data = response.json()
        # Validate response structure
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "step" in data and isinstance(data["step"], str) and data["step"], "step field must be a non-empty string"
        assert "estimated_time" in data, "estimated_time must be present"
        assert "next_action" in data and "onboarding" in data["next_action"].lower(), "next_action must contain onboarding instructions"
        # Optionally, validate against StepResponse schema
        try:
            StepResponse(**data)
        except Exception as e:
            pytest.fail(f"Response does not match StepResponse schema: {e}")

@pytest.mark.parametrize(
    "payload,missing_field",
    [
        ({}, "employee_id"),
        ({"employee_id": "EMP12345"}, "role"),
        ({"employee_id": "EMP12345", "role": "Software Engineer"}, "department"),
        ({"employee_id": "EMP12345", "role": "Software Engineer", "department": "Engineering"}, "operating_system"),
    ]
)
def test_start_onboarding_validation_error(client, payload, missing_field):
    """
    Functional test: Validates that /start returns 422 ValidationError if required fields are missing.
    """
    response = client.post("/start", json=payload)
    assert response.status_code == 422, f"Expected 422 ValidationError when '{missing_field}' is missing"

def test_start_onboarding_internal_error(client, valid_user_context):
    """
    Functional test: Validates that /start returns HTTP 500 if an internal error occurs.
    """
    # Patch the onboarding step generator to raise an exception
    with patch("app.routers.onboarding.generate_first_step", side_effect=Exception("Internal error")):
        response = client.post("/start", json=valid_user_context)
        assert response.status_code == 500, "Expected HTTP 500 on internal error"
        assert "error" in response.json() or "detail" in response.json(), "Error message should be present in response"

