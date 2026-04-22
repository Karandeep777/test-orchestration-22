
import pytest
from unittest.mock import patch, MagicMock

# Assume the following imports are from the actual codebase under test
# from hr_agent.document_processor import process_document_upload, UploadResponse, DocumentError

@pytest.fixture
def sample_document_text_missing_fields():
    """
    Fixture providing a sample document text missing 'employee_id' and 'start_date'.
    """
    return "This is a contract for John Doe. Position: Software Engineer."

@pytest.fixture
def expected_missing_fields():
    """
    Fixture providing the expected missing fields for the test.
    """
    return ['employee_id', 'start_date']

@pytest.fixture
def mock_field_extractor(expected_missing_fields):
    """
    Fixture that mocks the field extraction function to simulate missing fields.
    """
    def _extract_fields(document_text):
        # Simulate extraction missing 'employee_id' and 'start_date'
        return {
            'employee_name': 'John Doe',
            'position': 'Software Engineer'
            # 'employee_id' and 'start_date' are missing
        }
    return MagicMock(side_effect=_extract_fields)

@pytest.fixture
def mock_upload_response_class():
    """
    Fixture that returns a mock UploadResponse class for constructing responses.
    """
    class MockError:
        def __init__(self, error_code, message):
            self.error_code = error_code
            self.message = message

    class MockUploadResponse:
        def __init__(self, success, errors, flagged_fields):
            self.success = success
            self.errors = errors
            self.flagged_fields = flagged_fields

    return MockUploadResponse, MockError

def test_integration_end_to_end_document_processing_with_missing_fields(
    sample_document_text_missing_fields,
    expected_missing_fields,
    mock_field_extractor,
    mock_upload_response_class
):
    """
    Integration test: End-to-End Document Processing with Missing Fields.
    Tests that when a document is missing required HR fields ('employee_id', 'start_date'),
    the workflow flags the missing fields and includes them in the response.
    """
    # Patch the field extraction function used in the workflow
    # Patch the UploadResponse and DocumentError classes if needed
    # Assume process_document_upload is the main entry point

    # Import inside test to avoid import errors if not present in environment
    with patch('hr_agent.document_processor.extract_fields', mock_field_extractor):
        MockUploadResponse, MockError = mock_upload_response_class

        # Patch UploadResponse to use our mock class
        with patch('hr_agent.document_processor.UploadResponse', MockUploadResponse), \
             patch('hr_agent.document_processor.DocumentError', MockError):

            # Simulate the process_document_upload function
            # The actual function signature may differ; adjust as needed
            from hr_agent.document_processor import process_document_upload

            response = process_document_upload(document_text=sample_document_text_missing_fields)

            # Assertions per success_criteria
            assert response.success is True, "Response should indicate success"
            assert response.errors is not None, "Errors should not be None"
            assert len(response.errors) > 0, "There should be at least one error"
            assert response.errors[0].error_code == 'ERR_MISSING_FIELD', "Error code should indicate missing field"
            error_message = response.errors[0].message
            for field in expected_missing_fields:
                assert field in error_message, f"Error message should mention missing field: {field}"
            for field in expected_missing_fields:
                assert field in response.flagged_fields, f"Flagged fields should include missing field: {field}"
