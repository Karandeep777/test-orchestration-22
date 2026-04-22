
# config.py

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ITSetupGuideAgentConfig")

class ConfigError(Exception):
    pass

class AgentConfig:
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT",
        "You are the IT Setup Guide Agent, an interactive onboarding assistant for new employees. Always provide clear, numbered, step-by-step instructions tailored to the user's operating system, role, and department. After each step, ask: \"Done? Type YES to continue or describe the issue you're seeing.\" Never ask for or store passwords. Use encouraging language, estimate time remaining at milestones, and proactively include fixes for common errors. If an issue cannot be resolved, offer to create an IT helpdesk ticket with the employee's consent."
    )
    LLM_USER_PROMPT_TEMPLATE = os.getenv("LLM_USER_PROMPT_TEMPLATE",
        "Welcome to your IT onboarding! Let's get started. Please tell me your operating system (Windows 11, macOS Sonoma/Sequoia, or Ubuntu), your role, and your department. I'll guide you step-by-step through your setup."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        "I'm a developer on macOS Sonoma.",
        "I'm in Finance and using Windows 11."
    ]

    # API Keys and Endpoints
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    IT_TICKETING_API_URL = os.getenv("IT_TICKETING_API_URL")
    IT_TICKETING_API_TOKEN = os.getenv("IT_TICKETING_API_TOKEN")
    PROGRESS_PERSISTENCE_URL = os.getenv("PROGRESS_PERSISTENCE_URL")
    PROGRESS_PERSISTENCE_TOKEN = os.getenv("PROGRESS_PERSISTENCE_TOKEN")
    EMAIL_NOTIFICATION_API_URL = os.getenv("EMAIL_NOTIFICATION_API_URL")
    EMAIL_NOTIFICATION_API_TOKEN = os.getenv("EMAIL_NOTIFICATION_API_TOKEN")

    # Domain-specific settings
    DOMAIN = "general"
    AGENT_NAME = "IT Setup Guide Agent"
    DEFAULT_OS_LIST = ["Windows 11", "macOS Sonoma", "macOS Sequoia", "Ubuntu"]
    DEFAULT_ROLES = ["Developer", "Finance", "HR", "Engineering"]
    DEFAULT_DEPARTMENTS = ["Engineering", "Finance", "HR"]

    # API Requirements
    API_REQUIREMENTS = [
        {
            "name": "OpenAI API",
            "type": "external",
            "purpose": "LLM-based step generation and troubleshooting guidance",
            "authentication": "API Key (secure vault)",
            "rate_limits": "As per OpenAI subscription"
        },
        {
            "name": "Azure AI Search",
            "type": "external",
            "purpose": "Retrieve onboarding documentation and technical answers",
            "authentication": "OAuth2 / API Key",
            "rate_limits": "1000 requests/minute"
        },
        {
            "name": "IT Ticketing System",
            "type": "external",
            "purpose": "Auto-create and track IT helpdesk tickets",
            "authentication": "OAuth2 (service principal)",
            "rate_limits": "500 requests/hour"
        },
        {
            "name": "Progress Persistence Service",
            "type": "internal",
            "purpose": "Save and resume onboarding progress",
            "authentication": "Service-to-service token",
            "rate_limits": "1000 requests/minute"
        },
        {
            "name": "Email Notification Service",
            "type": "internal",
            "purpose": "Send completion certificates and notifications",
            "authentication": "Service-to-service token",
            "rate_limits": "500 emails/hour"
        }
    ]

    # Validation and error handling
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.AZURE_SEARCH_KEY:
            missing.append("AZURE_SEARCH_KEY")
        if not cls.AZURE_SEARCH_ENDPOINT:
            missing.append("AZURE_SEARCH_ENDPOINT")
        if not cls.IT_TICKETING_API_URL:
            missing.append("IT_TICKETING_API_URL")
        if not cls.IT_TICKETING_API_TOKEN:
            missing.append("IT_TICKETING_API_TOKEN")
        if not cls.PROGRESS_PERSISTENCE_URL:
            missing.append("PROGRESS_PERSISTENCE_URL")
        if not cls.PROGRESS_PERSISTENCE_TOKEN:
            missing.append("PROGRESS_PERSISTENCE_TOKEN")
        if not cls.EMAIL_NOTIFICATION_API_URL:
            missing.append("EMAIL_NOTIFICATION_API_URL")
        if not cls.EMAIL_NOTIFICATION_API_TOKEN:
            missing.append("EMAIL_NOTIFICATION_API_TOKEN")
        if missing:
            logger.error(f"Missing required API keys or endpoints: {', '.join(missing)}")
            raise ConfigError(f"Missing required API keys or endpoints: {', '.join(missing)}")
        return True

    # Default values and fallbacks
    @classmethod
    def get_llm_config(cls) -> Dict:
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_api_keys(cls) -> Dict:
        return {
            "openai_api_key": cls.OPENAI_API_KEY,
            "azure_search_key": cls.AZURE_SEARCH_KEY,
            "azure_search_endpoint": cls.AZURE_SEARCH_ENDPOINT,
            "it_ticketing_api_url": cls.IT_TICKETING_API_URL,
            "it_ticketing_api_token": cls.IT_TICKETING_API_TOKEN,
            "progress_persistence_url": cls.PROGRESS_PERSISTENCE_URL,
            "progress_persistence_token": cls.PROGRESS_PERSISTENCE_TOKEN,
            "email_notification_api_url": cls.EMAIL_NOTIFICATION_API_URL,
            "email_notification_api_token": cls.EMAIL_NOTIFICATION_API_TOKEN
        }

    @classmethod
    def get_domain_settings(cls) -> Dict:
        return {
            "domain": cls.DOMAIN,
            "agent_name": cls.AGENT_NAME,
            "default_os_list": cls.DEFAULT_OS_LIST,
            "default_roles": cls.DEFAULT_ROLES,
            "default_departments": cls.DEFAULT_DEPARTMENTS
        }

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
# domain_settings = AgentConfig.get_domain_settings()
