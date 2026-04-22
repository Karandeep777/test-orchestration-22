
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("CompanyPolicyQAAgentConfig")

class ConfigError(Exception):
    pass

class AgentConfig:
    """
    Central configuration management for Company Policy Q&A Agent.
    Handles environment variable loading, API key management, LLM config,
    domain settings, validation, and default values.
    """

    # --- API Keys & Endpoints ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    AZURE_AI_SEARCH_ENDPOINT: Optional[str] = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    AZURE_AI_SEARCH_KEY: Optional[str] = os.getenv("AZURE_AI_SEARCH_KEY")
    AZURE_AI_SEARCH_INDEX: str = os.getenv("AZURE_AI_SEARCH_INDEX", "policy-index")
    ESCALATION_API_URL: str = os.getenv("ESCALATION_API_URL", "http://localhost:9000/escalate")
    ESCALATION_API_KEY: Optional[str] = os.getenv("ESCALATION_API_KEY")
    LOGGING_API_URL: str = os.getenv("LOGGING_API_URL", "http://localhost:9000/log")
    LOGGING_API_KEY: Optional[str] = os.getenv("LOGGING_API_KEY")
    HR_ADMIN_API_URL: str = os.getenv("HR_ADMIN_API_URL", "http://localhost:9000/hradmin")
    HR_ADMIN_API_KEY: Optional[str] = os.getenv("HR_ADMIN_API_KEY")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- LLM Configuration ---
    LLM_CONFIG: Dict[str, Any] = {
        "provider": "openai",
        "model": os.getenv("LLM_MODEL", "gpt-4.1"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
        "system_prompt": os.getenv(
            "LLM_SYSTEM_PROMPT",
            "You are the Company Policy Q&A Agent. Your role is to answer employee questions about company policies, HR procedures, and handbook content using only the official, indexed documents. Every answer must cite the source document, section, and last-updated date. If you are not confident in your answer or the topic is sensitive or legal, escalate to the appropriate specialist. Never fabricate details or provide legal advice. Keep answers concise and professional, and offer more detail if requested."
        ),
        "user_prompt_template": os.getenv(
            "LLM_USER_PROMPT_TEMPLATE",
            "Please enter your question about company policies, HR procedures, or the employee handbook. For example: 'How many sick days do I get per year?' or 'What is the process for requesting parental leave?'"
        ),
        "few_shot_examples": [
            "Q: How many sick days do I get per year? A: According to Section 3.1 of the Leave Policy (updated Jan 2026), full-time employees are entitled to 10 paid sick days per year.",
            "Q: Can I work from home on Fridays? A: According to Section 5.2 of the Remote Work Policy (updated Mar 2026), employees may work from home on Fridays with manager approval.",
            "Q: What happens to my unvested stock if I resign? A: According to Section 7.4 of the Equity Policy (updated Feb 2026), unvested stock options are forfeited upon resignation. For further details or legal implications, I will connect you to HR/Legal.",
            "Q: What is the process for requesting parental leave? A: According to Section 4.3 of the Leave Policy (updated Jan 2026), to request parental leave, submit the Parental Leave Request Form to HR at least 30 days in advance. Would you like a link to the form?"
        ]
    }

    # --- Domain-Specific Settings ---
    DOMAIN: str = "human_resources"
    AGENT_NAME: str = "Company Policy Q&A Agent"
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24 hours
    KNOWLEDGE_BASE_DOCUMENT_LIST: List[str] = []
    try:
        import json
        _kb_env = os.getenv("KNOWLEDGE_BASE_DOCUMENT_LIST", "[]")
        KNOWLEDGE_BASE_DOCUMENT_LIST = json.loads(_kb_env)
    except Exception as e:
        logger.warning("Failed to parse KNOWLEDGE_BASE_DOCUMENT_LIST, using empty list.")

    # --- API Requirements ---
    API_REQUIREMENTS: List[Dict[str, Any]] = [
        {
            "name": "Azure AI Search API",
            "type": "external",
            "purpose": "Semantic search and retrieval of policy document chunks with metadata.",
            "authentication": "API Key or OAuth2",
            "rate_limits": "As per Azure subscription"
        },
        {
            "name": "OpenAI API",
            "type": "external",
            "purpose": "LLM inference for answer generation.",
            "authentication": "API Key",
            "rate_limits": "As per OpenAI subscription"
        },
        {
            "name": "Escalation Ticketing API",
            "type": "internal",
            "purpose": "Create and track escalation tickets for HR or Legal.",
            "authentication": "SSO/JWT",
            "rate_limits": "200 requests/minute"
        },
        {
            "name": "HR Admin Panel API",
            "type": "internal",
            "purpose": "Upload, update, or invalidate policy documents.",
            "authentication": "SSO/JWT",
            "rate_limits": "50 requests/minute"
        },
        {
            "name": "Logging API",
            "type": "internal",
            "purpose": "Log unanswered or ambiguous questions for analytics.",
            "authentication": "SSO/JWT",
            "rate_limits": "500 requests/minute"
        }
    ]

    # --- Validation ---
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.AZURE_AI_SEARCH_ENDPOINT:
            missing.append("AZURE_AI_SEARCH_ENDPOINT")
        if not cls.AZURE_AI_SEARCH_KEY:
            missing.append("AZURE_AI_SEARCH_KEY")
        if not cls.KNOWLEDGE_BASE_DOCUMENT_LIST or not isinstance(cls.KNOWLEDGE_BASE_DOCUMENT_LIST, list):
            missing.append("KNOWLEDGE_BASE_DOCUMENT_LIST")
        if missing:
            logger.error(f"Missing required configuration keys: {', '.join(missing)}")
            raise ConfigError(f"Missing required configuration keys: {', '.join(missing)}")

    # --- Fallbacks & Defaults ---
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return cls.LLM_CONFIG

    @classmethod
    def get_api_keys(cls) -> Dict[str, Optional[str]]:
        return {
            "openai": cls.OPENAI_API_KEY,
            "azure_ai_search": cls.AZURE_AI_SEARCH_KEY,
            "escalation": cls.ESCALATION_API_KEY,
            "logging": cls.LOGGING_API_KEY,
            "hr_admin": cls.HR_ADMIN_API_KEY
        }

    @classmethod
    def get_api_endpoints(cls) -> Dict[str, str]:
        return {
            "azure_ai_search_endpoint": cls.AZURE_AI_SEARCH_ENDPOINT,
            "azure_ai_search_index": cls.AZURE_AI_SEARCH_INDEX,
            "escalation_api_url": cls.ESCALATION_API_URL,
            "logging_api_url": cls.LOGGING_API_URL,
            "hr_admin_api_url": cls.HR_ADMIN_API_URL,
            "redis_url": cls.REDIS_URL
        }

    @classmethod
    def get_domain_settings(cls) -> Dict[str, Any]:
        return {
            "domain": cls.DOMAIN,
            "agent_name": cls.AGENT_NAME,
            "session_ttl_seconds": cls.SESSION_TTL_SECONDS,
            "knowledge_base_document_list": cls.KNOWLEDGE_BASE_DOCUMENT_LIST
        }

    @classmethod
    def get_api_requirements(cls) -> List[Dict[str, Any]]:
        return cls.API_REQUIREMENTS

# Validate configuration on import
try:
    AgentConfig.validate()
except ConfigError as e:
    logger.critical(str(e))
    raise

# Usage example (in other modules):
# from config import AgentConfig
# llm_config = AgentConfig.get_llm_config()
# api_keys = AgentConfig.get_api_keys()
# endpoints = AgentConfig.get_api_endpoints()
# domain_settings = AgentConfig.get_domain_settings()
# api_requirements = AgentConfig.get_api_requirements()
