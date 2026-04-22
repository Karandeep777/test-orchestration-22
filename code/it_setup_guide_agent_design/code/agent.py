

class Response:
    def __init__(self, **kwargs):
        self.status_code = 200
        self._data = kwargs
    def json(self):
        return self._data
import time as _time
try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': True,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 2,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, EmailStr
from dotenv import load_dotenv
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cryptography.fernet import Fernet, InvalidToken
import jwt
import re
import json

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ITSetupGuideAgent")

# Configuration Management
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY", "")
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    IT_TICKETING_API_URL: str = os.getenv("IT_TICKETING_API_URL", "")
    IT_TICKETING_API_TOKEN: str = os.getenv("IT_TICKETING_API_TOKEN", "")
    EMAIL_NOTIFICATION_API_URL: str = os.getenv("EMAIL_NOTIFICATION_API_URL", "")
    EMAIL_NOTIFICATION_API_TOKEN: str = os.getenv("EMAIL_NOTIFICATION_API_TOKEN", "")
    PROGRESS_PERSISTENCE_URL: str = os.getenv("PROGRESS_PERSISTENCE_URL", "")
    PROGRESS_PERSISTENCE_TOKEN: str = os.getenv("PROGRESS_PERSISTENCE_TOKEN", "")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    JWT_SECRET: str = os.getenv("JWT_SECRET", Fernet.generate_key().decode())
    JWT_ALGORITHM: str = "HS256"
    ALLOWED_ORIGINS: List[str] = ["*"]

    @classmethod
    def validate(cls):
        missing = []
        for key in [
            "OPENAI_API_KEY", "AZURE_SEARCH_KEY", "AZURE_SEARCH_ENDPOINT",
            "IT_TICKETING_API_URL", "IT_TICKETING_API_TOKEN",
            "EMAIL_NOTIFICATION_API_URL", "EMAIL_NOTIFICATION_API_TOKEN",
            "PROGRESS_PERSISTENCE_URL", "PROGRESS_PERSISTENCE_TOKEN",
            "ENCRYPTION_KEY", "JWT_SECRET"
        ]:
            if not getattr(cls, key):
                missing.append(key)
        if missing:
            logger.warning(f"Missing config keys: {missing}")
        return missing

Config.validate()

# Utility Functions
@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(data: str) -> str:
    # Mask emails and phone numbers
    data = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', data)
    data = re.sub(r'\b\d{3}[-.\s]??\d{2,4}[-.\s]??\d{4}\b', '[PHONE]', data)
    return data

def encrypt_data(data: str) -> str:
    _obs_t0 = _time.time()
    f = Fernet(Config.ENCRYPTION_KEY.encode())
    try:
        trace_tool_call(
            tool_name='ENCRYPTION_KEY.encode',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(f)[:200] if f is not None else None,
            status="success",
        )
    except Exception:
        pass
    _obs_t0 = _time.time()
    _obs_resp = f.encrypt(data.encode()).decode()
    try:
        trace_tool_call(
            tool_name='f.encrypt',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(_obs_resp)[:200] if _obs_resp is not None else None,
            status="success",
        )
    except Exception:
        pass
    return _obs_resp

def decrypt_data(token: str) -> str:
    _obs_t0 = _time.time()
    f = Fernet(Config.ENCRYPTION_KEY.encode())
    try:
        trace_tool_call(
            tool_name='ENCRYPTION_KEY.encode',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(f)[:200] if f is not None else None,
            status="success",
        )
    except Exception:
        pass
    _obs_t0 = _time.time()
    _obs_resp = f.decrypt(token.encode()).decode()
    try:
        trace_tool_call(
            tool_name='f.decrypt',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(_obs_resp)[:200] if _obs_resp is not None else None,
            status="success",
        )
    except Exception:
        pass
    return _obs_resp

def estimate_time_remaining(current_step: int, total_steps: int) -> str:
    avg_time_per_step = 2  # minutes, can be refined
    remaining = (total_steps - current_step) * avg_time_per_step
    return f"Estimated time remaining: {remaining} minutes."

# Pydantic Models
class UserContext(BaseModel):
    employee_id: str = Field(..., min_length=2, max_length=64)
    role: str = Field(..., min_length=2, max_length=64)
    department: str = Field(..., min_length=2, max_length=64)
    operating_system: str = Field(..., min_length=2, max_length=64)
    email: Optional[EmailStr] = None

    @field_validator("employee_id", "role", "department", "operating_system")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty.")
        if len(v) > 128:
            raise ValueError("Field too long.")
        return v

class StepInput(BaseModel):
    employee_id: str = Field(..., min_length=2, max_length=64)
    user_input: str = Field(..., min_length=1, max_length=50000)

    @field_validator("user_input")
    @classmethod
    def clean_input(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Input cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input too long (max 50,000 characters).")
        return v

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    error_message: str
    tips: Optional[str] = None

class StepResponse(BaseModel):
    success: bool = True
    step: str
    estimated_time: Optional[str] = None
    next_action: Optional[str] = None

class TicketResponse(BaseModel):
    success: bool = True
    ticket_id: str
    message: str

class CompletionResponse(BaseModel):
    success: bool = True
    message: str

# Security & Compliance Layer
class SecurityManager:
    def __init__(self):
        self.jwt_secret = Config.JWT_SECRET
        self.jwt_algorithm = Config.JWT_ALGORITHM

    def authenticate_user(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            logger.info(f"User authenticated: {payload.get('sub', 'unknown')}")
            return True
        except jwt.ExpiredSignatureError:
            logger.warning("JWT expired.")
            return False
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT.")
            return False

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def mask_pii(self, data: str) -> str:
        return mask_pii(data)

    def encrypt_data(self, data: str) -> str:
        return encrypt_data(data)

# Logging/Audit Layer
class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("AuditLogger")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        masked_details = {k: mask_pii(str(v)) for k, v in details.items()}
        self.logger.info(f"Event: {event_type} | Details: {masked_details}")

# Persistence Layer
class ProgressPersistenceService:
    def __init__(self):
        self.url = Config.PROGRESS_PERSISTENCE_URL
        self.token = Config.PROGRESS_PERSISTENCE_TOKEN
        self._store: Dict[str, Dict[str, Any]] = {}

    async def save_progress(self, employee_id: str, current_step: Dict[str, Any]) -> bool:
        # Simulate persistence (replace with actual API call)
        self._store[employee_id] = current_step
        logger.info(f"Progress saved for {employee_id}: {current_step}")
        return True

    async def load_progress(self, employee_id: str) -> Optional[Dict[str, Any]]:
        # Simulate load (replace with actual API call)
        progress = self._store.get(employee_id)
        logger.info(f"Progress loaded for {employee_id}: {progress}")
        return progress

# Session Management Layer
class ChatSessionManager:
    def __init__(self, persistence_service: ProgressPersistenceService, audit_logger: AuditLogger):
        self.persistence = persistence_service
        self.audit_logger = audit_logger

    async def start_session(self, employee_id: str, role: str, department: str, os: str) -> Dict[str, Any]:
        session = {
            "employee_id": employee_id,
            "role": role,
            "department": department,
            "operating_system": os,
            "current_step": 0,
            "completed": False
        }
        await self.persistence.save_progress(employee_id, session)
        self.audit_logger.log_event("SESSION_START", session)
        return session

    async def save_progress(self, employee_id: str, current_step: Dict[str, Any]) -> bool:
        result = await self.persistence.save_progress(employee_id, current_step)
        self.audit_logger.log_event("PROGRESS_SAVE", {"employee_id": employee_id, "current_step": current_step})
        return result

    async def resume_session(self, employee_id: str) -> Optional[Dict[str, Any]]:
        session = await self.persistence.load_progress(employee_id)
        self.audit_logger.log_event("SESSION_RESUME", {"employee_id": employee_id, "session": session})
        return session

# Integration Layer
class ToolIntegrationManager:
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(Exception))
    async def create_it_ticket(self, employee_id: str, error_details: str, consent: bool) -> Optional[str]:
        if not consent:
            self.audit_logger.log_event("TICKET_NOT_CREATED_NO_CONSENT", {"employee_id": employee_id})
            return None
        # Simulate ticket creation (replace with actual API call)
        ticket_id = f"TICKET-{employee_id}-{os.urandom(3).hex()}"
        self.audit_logger.log_event("TICKET_CREATED", {"employee_id": employee_id, "ticket_id": ticket_id, "error_details": error_details})
        return ticket_id

    async def send_completion_email(self, employee_id: str, summary: str) -> bool:
        # Simulate email sending (replace with actual API call)
        self.audit_logger.log_event("EMAIL_SENT", {"employee_id": employee_id, "summary": summary})
        return True

    async def save_progress(self, employee_id: str, current_step: Dict[str, Any]) -> bool:
        # This is a pass-through to ProgressPersistenceService in this design
        return True

# Knowledge Search Layer
class KnowledgeRetriever:
    def __init__(self):
        self.endpoint = Config.AZURE_SEARCH_ENDPOINT
        self.key = Config.AZURE_SEARCH_KEY

    async def search_knowledge(self, user_query: str) -> str:
        # Simulate Azure AI Search (replace with actual API call)
        # For demo, return a canned answer
        if "vpn" in user_query.lower():
            return "To set up VPN, open your system settings, navigate to Network, and add a new VPN connection. For detailed steps, visit the company VPN guide."
        return "No direct match found in documentation. Please describe your issue for further assistance."

# Business Rules Engine
class BusinessRulesEngine:
    def __init__(self):
        self.approved_software = {
            "Engineering": ["VS Code", "Docker", "Git", "Python"],
            "Finance": ["Excel", "QuickBooks"],
            "HR": ["Workday", "DocuSign"]
        }

    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # Implement only the rules needed for demo
        if rule_id == "BR-001-01":  # No Password Storage
            if "password" in context.get("user_input", "").lower():
                return False, "ACCESS_DENIED"
            return True, None
        if rule_id == "BR-002-01":  # Approved Software Only
            software = context.get("software_name", "")
            dept = context.get("department", "")
            if software and dept:
                if software not in self.approved_software.get(dept, []):
                    return False, "SOFTWARE_INSTALL_FAIL"
            return True, None
        if rule_id == "BR-003-01":  # Consent for Ticket Logging
            if not context.get("employee_consent", False):
                return False, "TICKET_CREATION_FAIL"
            return True, None
        if rule_id == "BR-004-01":  # Save Progress
            if not context.get("employee_id") or not context.get("current_step"):
                return False, "PROGRESS_SAVE_FAIL"
            return True, None
        return True, None

    def apply_decision_table(self, table_id: str, inputs: Dict[str, str]) -> List[str]:
        # Only DT-001 implemented
        if table_id == "DT-001":
            role = inputs.get("role", "")
            dept = inputs.get("department", "")
            if role.lower() == "developer" and dept.lower() == "engineering":
                return ["VS Code", "Docker", "Git", "Python"]
            if role.lower() == "finance" and dept.lower() == "finance":
                return ["Excel", "QuickBooks"]
            if role.lower() == "hr" and dept.lower() == "hr":
                return ["Workday", "DocuSign"]
        return []

# LLM Integration Layer
class LLMService:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4-1106-preview"  # Use GPT-4.1 if available, fallback to gpt-3.5-turbo
        self.fallback_model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.system_prompt = (
            "You are the IT Setup Guide Agent, an interactive onboarding assistant for new employees. "
            "Always provide clear, numbered, step-by-step instructions tailored to the user's operating system, role, and department. "
            "After each step, ask: \"Done? Type YES to continue or describe the issue you're seeing.\" "
            "Never ask for or store passwords. Use encouraging language, estimate time remaining at milestones, "
            "and proactively include fixes for common errors. If an issue cannot be resolved, offer to create an IT helpdesk ticket with the employee's consent."
        )
        self.user_prompt_template = (
            "Welcome to your IT onboarding! Let's get started. Please tell me your operating system (Windows 11, macOS Sonoma/Sequoia, or Ubuntu), your role, and your department. "
            "I'll guide you step-by-step through your setup."
        )
        self.few_shot_examples = [
            {"role": "user", "content": "I'm a developer on macOS Sonoma."},
            {"role": "user", "content": "I'm in Finance and using Windows 11."}
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(Exception))
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm(self, prompt: str, context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        for ex in self.few_shot_examples:
            messages.append(ex)
        messages.append({"role": "user", "content": prompt})
        try:
            _obs_t0 = _time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            try:
                trace_model_call(
                    provider='azure',
                    model_name=(getattr(self, "model", None) or getattr(getattr(self, "config", None), "model", None) or "unknown"),
                    prompt_tokens=(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0),
                    completion_tokens=(getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0),
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                )
            except Exception:
                pass
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM call failed: {e}, falling back to {self.fallback_model}")
            return await self.fallback_call_llm(prompt, context)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def fallback_call_llm(self, prompt: str, context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        for ex in self.few_shot_examples:
            messages.append(ex)
        messages.append({"role": "user", "content": prompt})
        _obs_t0 = _time.time()
        response = await self.client.chat.completions.create(
            model=self.fallback_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        try:
            trace_model_call(
                provider='azure',
                model_name=(getattr(self, "model", None) or getattr(getattr(self, "config", None), "model", None) or "unknown"),
                prompt_tokens=(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0),
                completion_tokens=(getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0),
                latency_ms=int((_time.time() - _obs_t0) * 1000),
            )
        except Exception:
            pass
        return response.choices[0].message.content

# Orchestration Layer
class StepOrchestrator:
    def __init__(
        self,
        llm_service: LLMService,
        business_rules: BusinessRulesEngine,
        knowledge_retriever: KnowledgeRetriever,
        session_manager: ChatSessionManager
    ):
        self.llm_service = llm_service
        self.business_rules = business_rules
        self.knowledge_retriever = knowledge_retriever
        self.session_manager = session_manager

    async def generate_next_step(self, context: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        # Compose prompt for LLM
        prompt = (
            f"Employee ID: {context.get('employee_id')}\n"
            f"Role: {context.get('role')}\n"
            f"Department: {context.get('department')}\n"
            f"Operating System: {context.get('operating_system')}\n"
            f"Current Step: {context.get('current_step', 0)}\n"
            f"Please generate the next onboarding step. "
            f"Include only approved software for this department. "
            f"Estimate time remaining if possible."
        )
        step = await self.llm_service.call_llm(prompt, context)
        # Estimate time remaining
        total_steps = 5  # For demo, assume 5 steps
        current_step = context.get("current_step", 0) + 1
        estimated_time = estimate_time_remaining(current_step, total_steps)
        return step, estimated_time

    @trace_agent(agent_name='IT Setup Guide Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def validate_step_completion(self, user_input: str) -> Tuple[bool, Optional[str]]:
        # Use LLM for sentiment/intent detection
        prompt = (
            f"User responded: '{user_input}'. "
            "Is the user confirming completion of the step (YES), or describing an issue? "
            "Reply with 'YES' or 'ISSUE' only."
        )
        response = await self.llm_service.call_llm(prompt, {})
        if "yes" in response.lower():
            return True, None
        return False, response

    async def handle_step_error(self, error_code: str) -> str:
        # Map error code to user-friendly message
        error_map = {
            "ACCESS_DENIED": "Sorry, I cannot assist with password-related steps. Please enter your password directly into the system dialog.",
            "SOFTWARE_INSTALL_FAIL": "The software requested is not approved for your department. Please contact IT if you believe this is an error.",
            "VPN_SETUP_ERROR": "There was an issue setting up VPN. Please check your network settings or contact IT.",
            "TICKET_CREATION_FAIL": "Unable to create an IT helpdesk ticket at this time. Please try again later.",
            "PROGRESS_SAVE_FAIL": "Could not save your progress. Please try again or contact support."
        }
        return error_map.get(error_code, "An unknown error occurred. Please contact IT support.")

# Main Agent Class
class ITSetupGuideAgent:
    def __init__(self):
        # Layer initialization
        self.security_manager = SecurityManager()
        self.audit_logger = AuditLogger()
        self.persistence_service = ProgressPersistenceService()
        self.session_manager = ChatSessionManager(self.persistence_service, self.audit_logger)
        self.tool_integration = ToolIntegrationManager(self.audit_logger)
        self.knowledge_retriever = KnowledgeRetriever()
        self.business_rules = BusinessRulesEngine()
        self.llm_service = LLMService()
        self.step_orchestrator = StepOrchestrator(
            self.llm_service,
            self.business_rules,
            self.knowledge_retriever,
            self.session_manager
        )

    async def start_onboarding(self, context: UserContext) -> StepResponse:
        # Start session and generate first step
        session = await self.session_manager.start_session(
            context.employee_id, context.role, context.department, context.operating_system
        )
        # Apply decision table for software checklist
        approved_software = self.business_rules.apply_decision_table(
            "DT-001", {"role": context.role, "department": context.department}
        )
        session["approved_software_list"] = approved_software
        await self.session_manager.save_progress(context.employee_id, session)
        step, estimated_time = await self.step_orchestrator.generate_next_step(session)
        self.audit_logger.log_event("ONBOARDING_STARTED", {"employee_id": context.employee_id, "step": step})
        return StepResponse(step=step, estimated_time=estimated_time, next_action="Please follow the instructions and reply 'YES' when done or describe any issue.")

    @trace_agent(agent_name='IT Setup Guide Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_step(self, step_input: StepInput) -> Union[StepResponse, ErrorResponse]:
        # Resume session
        session = await self.session_manager.resume_session(step_input.employee_id)
        if not session:
            return ErrorResponse(
                error_type="SESSION_NOT_FOUND",
                error_message="No active session found. Please start onboarding.",
                tips="Start a new session by providing your employee ID, role, department, and operating system."
            )
        # Business rule: No password storage
        ok, error_code = self.business_rules.evaluate_rule("BR-001-01", {"user_input": step_input.user_input})
        if not ok:
            error_msg = await self.step_orchestrator.handle_step_error(error_code)
            return ErrorResponse(
                error_type=error_code,
                error_message=error_msg,
                tips="Never share your password with anyone, including this agent."
            )
        # Validate step completion
        completed, issue = await self.step_orchestrator.validate_step_completion(step_input.user_input)
        if completed:
            session["current_step"] += 1
            if session["current_step"] >= 5:
                session["completed"] = True
                await self.session_manager.save_progress(step_input.employee_id, session)
                await self.tool_integration.send_completion_email(step_input.employee_id, "Onboarding completed successfully.")
                self.audit_logger.log_event("ONBOARDING_COMPLETED", {"employee_id": step_input.employee_id})
                return CompletionResponse(message="Congratulations! Your IT onboarding is complete. A completion certificate has been sent to your email.")
            else:
                await self.session_manager.save_progress(step_input.employee_id, session)
                step, estimated_time = await self.step_orchestrator.generate_next_step(session)
                return StepResponse(step=step, estimated_time=estimated_time, next_action="Please follow the instructions and reply 'YES' when done or describe any issue.")
        else:
            # Try to retrieve knowledge base answer
            kb_answer = await self.knowledge_retriever.search_knowledge(step_input.user_input)
            return ErrorResponse(
                error_type="STEP_ISSUE",
                error_message=f"I see you're having an issue: {issue}\n\n{kb_answer}",
                tips="If the issue persists, reply 'HELP' to create an IT helpdesk ticket."
            )

    async def create_ticket(self, employee_id: str, error_details: str, consent: bool) -> Union[TicketResponse, ErrorResponse]:
        ok, error_code = self.business_rules.evaluate_rule("BR-003-01", {"employee_consent": consent, "error_details": error_details})
        if not ok:
            error_msg = await self.step_orchestrator.handle_step_error(error_code)
            return ErrorResponse(
                error_type=error_code,
                error_message=error_msg,
                tips="Please provide consent to create a helpdesk ticket."
            )
        ticket_id = await self.tool_integration.create_it_ticket(employee_id, error_details, consent)
        if ticket_id:
            return TicketResponse(ticket_id=ticket_id, message="IT helpdesk ticket created successfully.")
        else:
            return ErrorResponse(
                error_type="TICKET_CREATION_FAIL",
                error_message="Failed to create IT helpdesk ticket.",
                tips="Try again later or contact IT support directly."
            )

# FastAPI App
app = FastAPI(
    title="IT Setup Guide Agent",
    description="Interactive IT onboarding assistant for new employees.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ITSetupGuideAgent()

# Exception Handlers
@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_type="VALIDATION_ERROR",
            error_message="Invalid input data. Please check your request for missing fields, incorrect types, or formatting issues.",
            tips="Ensure all required fields are present, use double quotes for JSON strings, and check for trailing commas."
        ).model_dump()
    )

@app.exception_handler(json.decoder.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    logger.error(f"Malformed JSON: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_type="MALFORMED_JSON",
            error_message="Malformed JSON in request body.",
            tips="Check for missing quotes, commas, or brackets. Use a JSON validator before submitting."
        ).model_dump()
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type="INTERNAL_SERVER_ERROR",
            error_message="An unexpected error occurred.",
            tips="Try again later or contact IT support."
        ).model_dump()
    )

# API Endpoints
@app.post("/start", response_model=StepResponse, responses={422: {"model": ErrorResponse}})
async def start_onboarding(context: UserContext):
    """
    Start a new IT onboarding session.
    """
    try:
        return await agent.start_onboarding(context)
    except ValidationError as ve:
        logger.error(f"Validation error in /start: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in /start: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/step", response_model=Union[StepResponse, ErrorResponse], responses={422: {"model": ErrorResponse}})
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_step(step_input: StepInput):
    """
    Process a user's response to the current onboarding step.
    """
    try:
        return await agent.process_step(step_input)
    except ValidationError as ve:
        logger.error(f"Validation error in /step: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in /step: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

class TicketRequest(BaseModel):
    employee_id: str = Field(..., min_length=2, max_length=64)
    error_details: str = Field(..., min_length=1, max_length=50000)
    consent: bool = Field(...)

    @field_validator("error_details")
    @classmethod
    def clean_error_details(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Error details cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Error details too long (max 50,000 characters).")
        return v

@app.post("/ticket", response_model=Union[TicketResponse, ErrorResponse], responses={422: {"model": ErrorResponse}})
@with_content_safety(config=GUARDRAILS_CONFIG)
async def create_ticket(request: TicketRequest):
    """
    Create an IT helpdesk ticket for unresolved onboarding issues.
    """
    try:
        return await agent.create_ticket(request.employee_id, request.error_details, request.consent)
    except ValidationError as ve:
        logger.error(f"Validation error in /ticket: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in /ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"success": True, "status": "ok"}

# Main entry point


async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting IT Setup Guide Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())