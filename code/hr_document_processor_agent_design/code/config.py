
# config.py for HR Document Processor Agent

import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AgentConfig:
    # 1. Environment variable loading
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_AI_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    AZURE_AI_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
    AUDIT_LOG_SERVICE_ACCOUNT = os.getenv("AUDIT_LOG_SERVICE_ACCOUNT")
    # Fallbacks for role access control
    ROLE_ACCESS_CONTROL = {
        "user": ["upload"],
        "hr_manager": ["upload", "batch_process"],
        "admin": ["upload", "batch_process", "audit"]
    }

    # 2. API key management & validation
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.AZURE_AI_SEARCH_ENDPOINT:
            missing.append("AZURE_AI_SEARCH_ENDPOINT")
        if not cls.AZURE_AI_SEARCH_KEY:
            missing.append("AZURE_AI_SEARCH_KEY")
        if missing:
            raise ConfigError(f"Missing required API keys or endpoints: {', '.join(missing)}")

    # 3. LLM configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are an HR Document Processor Agent. Your role is to automate the reading, classification, "
            "and extraction of structured data from employee onboarding documents. Always mask sensitive information, "
            "flag low-confidence extractions for HR review, and provide clear, concise summaries. Confirm document type "
            "before extraction and ensure all required documents are present. Adhere strictly to GDPR and CCPA compliance."
        ),
        "user_prompt_template": (
            "Please upload your onboarding document(s). Supported formats: PDF, DOCX, PNG, JPG. The agent will extract "
            "required HR fields, flag missing or incomplete data, and generate a summary for HR review."
        ),
        "few_shot_examples": [
            "Upload: Signed offer letter.pdf -> Document type: Offer Letter (confidence: 0.98)...",
            "Upload: NDA_scan.jpg -> Document type: NDA (confidence: 0.93)...",
            "Upload: W-4_form.pdf -> Document type: W-4 (confidence: 0.97)..."
        ]
    }

    # 4. Domain-specific settings
    DOMAIN = "human_resources"
    SUPPORTED_FORMATS = ["pdf", "docx", "png", "jpg", "jpeg"]
    REQUIRED_DOCS = [
        "offer_letter", "W-4", "I-9", "NDA", "employment_contract", "personal_info_form"
    ]
    REQUIRED_FIELDS = [
        "full_name", "employee_id", "start_date", "role", "department",
        "location", "salary_band", "tax_withholding_info", "emergency_contact"
    ]
    CONFIDENCE_THRESHOLD = 0.85
    FIELD_CONFIDENCE_MIN = 0.65
    AUDIT_LOG_RETENTION_DAYS = 90

    # 5. API requirements
    API_REQUIREMENTS = [
        {
            "name": "Document Upload API",
            "type": "external",
            "purpose": "Accepts onboarding document uploads from users",
            "authentication": "OAuth2 with SSO",
            "rate_limits": "20 requests/min/user"
        },
        {
            "name": "Batch Processing API",
            "type": "internal",
            "purpose": "Allows HR managers to process multiple documents",
            "authentication": "OAuth2 with role-based access",
            "rate_limits": "5 batch requests/min/manager"
        },
        {
            "name": "Azure AI Search API",
            "type": "external",
            "purpose": "Retrieves HR knowledge base content",
            "authentication": "API Key",
            "rate_limits": "1000 requests/day"
        },
        {
            "name": "Audit Log API",
            "type": "internal",
            "purpose": "Records all document access and extraction events",
            "authentication": "Service account",
            "rate_limits": "Unlimited (subject to log storage constraints)"
        }
    ]

    # 6. Validation and error handling
    ERROR_CODES = [
        "ERR_MISSING_FIELD",
        "ERR_UNSUPPORTED_FORMAT",
        "ERR_DUPLICATE_DOCUMENT",
        "ERR_LOW_CONFIDENCE",
        "ERR_ACCESS_DENIED",
        "ERR_MISSING_DOCUMENT"
    ]

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return cls.LLM_CONFIG

    @classmethod
    def get_api_key(cls, service: str) -> str:
        if service == "openai":
            if not cls.OPENAI_API_KEY:
                raise ConfigError("Missing OpenAI API Key")
            return cls.OPENAI_API_KEY
        elif service == "azure_ai_search":
            if not cls.AZURE_AI_SEARCH_KEY:
                raise ConfigError("Missing Azure AI Search API Key")
            return cls.AZURE_AI_SEARCH_KEY
        elif service == "audit_log":
            if not cls.AUDIT_LOG_SERVICE_ACCOUNT:
                raise ConfigError("Missing Audit Log Service Account")
            return cls.AUDIT_LOG_SERVICE_ACCOUNT
        else:
            raise ConfigError(f"Unknown service: {service}")

    # 7. Default values and fallbacks
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        return cls.SUPPORTED_FORMATS

    @classmethod
    def get_required_docs(cls) -> List[str]:
        return cls.REQUIRED_DOCS

    @classmethod
    def get_required_fields(cls) -> List[str]:
        return cls.REQUIRED_FIELDS

    @classmethod
    def get_role_access_control(cls) -> Dict[str, List[str]]:
        return cls.ROLE_ACCESS_CONTROL

    @classmethod
    def get_confidence_threshold(cls) -> float:
        return cls.CONFIDENCE_THRESHOLD

    @classmethod
    def get_field_confidence_min(cls) -> float:
        return cls.FIELD_CONFIDENCE_MIN

    @classmethod
    def get_audit_log_retention_days(cls) -> int:
        return cls.AUDIT_LOG_RETENTION_DAYS

    @classmethod
    def get_domain(cls) -> str:
        return cls.DOMAIN

    @classmethod
    def get_api_requirements(cls) -> List[Dict[str, Any]]:
        return cls.API_REQUIREMENTS

    @classmethod
    def get_error_codes(cls) -> List[str]:
        return cls.ERROR_CODES

    @classmethod
    def check_all(cls):
        try:
            cls.validate()
        except ConfigError as e:
            logging.error(f"Agent configuration error: {str(e)}")
            raise

# Validate configuration at import
try:
    AgentConfig.check_all()
except ConfigError as e:
    # Commented out: raise SystemExit(str(e))
    logging.error(f"Startup configuration error: {str(e)}")

# Usage example (commented out):
# llm_cfg = AgentConfig.get_llm_config()
# openai_key = AgentConfig.get_api_key("openai")
# formats = AgentConfig.get_supported_formats()
