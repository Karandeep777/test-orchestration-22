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
import re
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple, Union
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, constr
from dotenv import load_dotenv

import openai
from openai import AsyncOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import redis
import nltk
import spacy

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("CompanyPolicyQAAgent")

# --- Configuration Management ---

class Config:
    """Configuration loader and validator for the agent."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    AZURE_AI_SEARCH_ENDPOINT: str = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    AZURE_AI_SEARCH_KEY: str = os.getenv("AZURE_AI_SEARCH_KEY")
    AZURE_AI_SEARCH_INDEX: str = os.getenv("AZURE_AI_SEARCH_INDEX", "policy-index")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ESCALATION_API_URL: str = os.getenv("ESCALATION_API_URL", "http://localhost:9000/escalate")
    ESCALATION_API_KEY: str = os.getenv("ESCALATION_API_KEY", "")
    LOGGING_API_URL: str = os.getenv("LOGGING_API_URL", "http://localhost:9000/log")
    LOGGING_API_KEY: str = os.getenv("LOGGING_API_KEY", "")
    HR_ADMIN_API_URL: str = os.getenv("HR_ADMIN_API_URL", "http://localhost:9000/hradmin")
    HR_ADMIN_API_KEY: str = os.getenv("HR_ADMIN_API_KEY", "")
    KNOWLEDGE_BASE_DOCUMENT_LIST: str = os.getenv("KNOWLEDGE_BASE_DOCUMENT_LIST", "[]")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24 hours

    @classmethod
    def validate(cls):
        missing = []
        for key in [
            "OPENAI_API_KEY", "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_KEY",
            "KNOWLEDGE_BASE_DOCUMENT_LIST"
        ]:
            if not getattr(cls, key):
                missing.append(key)
        if missing:
            raise RuntimeError(f"Missing required configuration keys: {', '.join(missing)}")

Config.validate()

# --- Pydantic Models ---

class UserQuery(BaseModel):
    user_id: constr(strip_whitespace=True, min_length=1, max_length=128)
    query: constr(strip_whitespace=True, min_length=1, max_length=50000)

    @field_validator("query")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def clean_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Query exceeds 50,000 character limit.")
        # Remove dangerous characters, excessive whitespace
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v)
        return v

class ProcessedQuery(BaseModel):
    user_id: str
    sanitized_query: str
    session_context: Optional[Dict[str, Any]] = None

class PolicyChunk(BaseModel):
    content: str
    source: str
    section: str
    last_updated: str
    metadata: Dict[str, Any]

class LLMInput(BaseModel):
    system_prompt: str
    user_prompt: str
    context_chunks: List[PolicyChunk]
    few_shot_examples: List[str]
    session_context: Optional[Dict[str, Any]] = None

class LLMOutput(BaseModel):
    answer: str
    confidence_score: float
    cited_source: Optional[str] = None
    cited_section: Optional[str] = None
    cited_last_updated: Optional[str] = None
    detected_topic: Optional[str] = None

class RuleResult(BaseModel):
    answer: str
    confidence_score: float
    cited_source: Optional[str]
    cited_section: Optional[str]
    cited_last_updated: Optional[str]
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    disclaimer: Optional[str] = None
    detected_topic: Optional[str] = None

class FormattedResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    cited_source: Optional[str] = None
    cited_section: Optional[str] = None
    cited_last_updated: Optional[str] = None
    disclaimer: Optional[str] = None
    escalation_ticket_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    fixing_tips: Optional[str] = None

# --- Utility: PII Redactor ---

class PIIRedactor:
    """Detects and redacts PII using regex and spaCy NER."""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Download if not present
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        # Basic regex for emails, phones, SSN, etc.
        self.regex_patterns = [
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
            (re.compile(r"\b\d{10}\b"), "[REDACTED_PHONE]"),
            (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[REDACTED_EMAIL]"),
        ]

    @trace_agent(agent_name='Company Policy Q&A Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def redact(self, text: str) -> str:
        for pattern, repl in self.regex_patterns:
            text = pattern.sub(repl, text)
        doc = self.nlp(text)
        redacted = text
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "ORG", "DATE", "LOC"}:
                redacted = redacted.replace(ent.text, f"[REDACTED_{ent.label_}]")
        return redacted

# --- Utility: Session Manager ---

class SessionManager:
    """Manages conversational context and session state using Redis."""
    def __init__(self, redis_url: str, ttl_seconds: int = 86400):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def get_session(self, user_id: str) -> Dict[str, Any]:
        session = self.redis.get(f"session:{user_id}")
        if session:
            try:
                return json.loads(session)
            except Exception:
                return {}
        return {}

    def update_session(self, user_id: str, context: Dict[str, Any]):
        self.redis.setex(f"session:{user_id}", self.ttl, json.dumps(context))

# --- Query Processor ---

class QueryProcessor:
    """Receives, sanitizes, and preprocesses user queries; manages session context."""
    def __init__(self, session_manager: SessionManager, pii_redactor: PIIRedactor):
        self.session_manager = session_manager
        self.pii_redactor = pii_redactor

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def process_query(self, input_query: UserQuery) -> ProcessedQuery:
        sanitized = self.pii_redactor.redact(input_query.query)
        session_context = self.session_manager.get_session(input_query.user_id)
        return ProcessedQuery(
            user_id=input_query.user_id,
            sanitized_query=sanitized,
            session_context=session_context
        )

# --- Azure AI Search Connector ---

class AzureAISearchConnector:
    """Performs semantic search over indexed policy documents and retrieves relevant chunks with metadata."""
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key)
        )

    @trace_agent(agent_name='Company Policy Q&A Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def search_documents(self, query: ProcessedQuery) -> List[PolicyChunk]:
        try:
            results = self.client.search(
                search_text=query.sanitized_query,
                top=3,
                query_type="semantic"
            )
            chunks = []
            for doc in results:
                chunks.append(PolicyChunk(
                    content=doc.get("content", ""),
                    source=doc.get("source", ""),
                    section=doc.get("section", ""),
                    last_updated=doc.get("last_updated", ""),
                    metadata={k: v for k, v in doc.items() if k not in {"content", "source", "section", "last_updated"}}
                ))
            return chunks
        except Exception as e:
            logger.error(f"Azure AI Search error: {e}")
            return []

# --- LLM Orchestrator ---

class LLMOrchestrator:
    """Constructs prompts, invokes LLM, applies few-shot examples, and manages fallback to alternate models."""
    def __init__(self, openai_api_key: str, config: Dict[str, Any]):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = config.get("model", "gpt-4.1")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2000)
        self.system_prompt = config.get("system_prompt")
        self.few_shot_examples = config.get("few_shot_examples", [])
        self.fallback_model = "gpt-3.5-turbo"

    @trace_agent(agent_name='Company Policy Q&A Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, prompt: LLMInput) -> LLMOutput:
        context_text = "\n\n".join(
            [f"Source: {c.source}, Section: {c.section}, Last Updated: {c.last_updated}\n{c.content}" for c in prompt.context_chunks]
        )
        few_shot = "\n".join(self.few_shot_examples)
        user_prompt = f"{prompt.user_prompt}\n\nContext:\n{context_text}\n\nExamples:\n{few_shot}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
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
            answer = response.choices[0].message.content.strip()
            # Confidence estimation: ask LLM to self-rate, or use heuristics
            confidence_score, cited_source, cited_section, cited_last_updated, detected_topic = \
                self._extract_metadata(answer, prompt.context_chunks)
            return LLMOutput(
                answer=answer,
                confidence_score=confidence_score,
                cited_source=cited_source,
                cited_section=cited_section,
                cited_last_updated=cited_last_updated,
                detected_topic=detected_topic
            )
        except Exception as e:
            logger.warning(f"LLM primary model failed: {e}, attempting fallback.")
            # Fallback to alternate model
            try:
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
                answer = response.choices[0].message.content.strip()
                confidence_score, cited_source, cited_section, cited_last_updated, detected_topic = \
                    self._extract_metadata(answer, prompt.context_chunks)
                return LLMOutput(
                    answer=answer,
                    confidence_score=confidence_score,
                    cited_source=cited_source,
                    cited_section=cited_section,
                    cited_last_updated=cited_last_updated,
                    detected_topic=detected_topic
                )
            except Exception as e2:
                logger.error(f"LLM fallback model failed: {e2}")
                raise RuntimeError("LLM inference failed for both primary and fallback models.")

    def _extract_metadata(self, answer: str, context_chunks: List[PolicyChunk]) -> Tuple[float, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Heuristic extraction of confidence, citation, and topic.
        """
        # Confidence: If answer contains explicit citation and matches context, high confidence.
        cited_source, cited_section, cited_last_updated = None, None, None
        for chunk in context_chunks:
            if chunk.source and chunk.section and chunk.last_updated:
                if chunk.section in answer and chunk.source in answer:
                    cited_source = chunk.source
                    cited_section = chunk.section
                    cited_last_updated = chunk.last_updated
                    break
        confidence_score = 0.9 if cited_source else 0.5 if any(chunk.source in answer for chunk in context_chunks) else 0.3
        # Topic detection: ask LLM or use keyword heuristics
        detected_topic = self._detect_topic(answer)
        return confidence_score, cited_source, cited_section, cited_last_updated, detected_topic

    def _detect_topic(self, answer: str) -> str:
        # Simple keyword-based topic detection
        topics = {
            "benefits": ["benefit", "insurance", "health", "dental", "vision"],
            "legal": ["legal", "liability", "law", "attorney", "court", "compliance"],
            "payroll": ["payroll", "salary", "wage", "pay", "compensation"],
            "termination": ["termination", "fired", "dismissal", "layoff"],
            "harassment": ["harassment", "bullying", "discrimination"],
            "disability": ["disability", "accommodation", "ADA"],
            "general": []
        }
        answer_lower = answer.lower()
        for topic, keywords in topics.items():
            if any(word in answer_lower for word in keywords):
                return topic
        return "general"

# --- Business Rule Engine ---

class BusinessRuleEngine:
    """Applies business rules for source citation, confidence thresholds, escalation, and compliance."""
    def __init__(self):
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.5
        }
        self.sensitive_topics = {"termination", "harassment", "disability"}
        self.legal_topics = {"legal"}

    def apply_rules(self, llm_output: LLMOutput, context: ProcessedQuery) -> RuleResult:
        # Rule: Source citation enforcement
        if not (llm_output.cited_source and llm_output.cited_section and llm_output.cited_last_updated):
            logger.warning("Source citation missing in answer.")
            return RuleResult(
                answer="I'm unable to provide an answer with a proper source citation. Your question will be escalated to HR.",
                confidence_score=0.0,
                cited_source=None,
                cited_section=None,
                cited_last_updated=None,
                escalation_required=True,
                escalation_reason="SOURCE_NOT_AVAILABLE",
                disclaimer=None,
                detected_topic=llm_output.detected_topic
            )
        # Rule: Confidence-based answering
        if llm_output.confidence_score >= self.confidence_thresholds["high"]:
            return RuleResult(
                answer=llm_output.answer,
                confidence_score=llm_output.confidence_score,
                cited_source=llm_output.cited_source,
                cited_section=llm_output.cited_section,
                cited_last_updated=llm_output.cited_last_updated,
                escalation_required=False,
                disclaimer=None,
                detected_topic=llm_output.detected_topic
            )
        elif llm_output.confidence_score >= self.confidence_thresholds["medium"]:
            disclaimer = "This answer is based on available policy information and may require confirmation."
            return RuleResult(
                answer=llm_output.answer,
                confidence_score=llm_output.confidence_score,
                cited_source=llm_output.cited_source,
                cited_section=llm_output.cited_section,
                cited_last_updated=llm_output.cited_last_updated,
                escalation_required=False,
                disclaimer=disclaimer,
                detected_topic=llm_output.detected_topic
            )
        else:
            return RuleResult(
                answer="I'm not confident in my answer. Your question will be escalated to HR for further assistance.",
                confidence_score=llm_output.confidence_score,
                cited_source=llm_output.cited_source,
                cited_section=llm_output.cited_section,
                cited_last_updated=llm_output.cited_last_updated,
                escalation_required=True,
                escalation_reason="NO_ANSWER_FOUND",
                disclaimer=None,
                detected_topic=llm_output.detected_topic
            )
        # Rule: Sensitive topic escalation
        # (Handled below for clarity)
        # Rule: Legal question routing
        # (Handled below for clarity)

    def check_sensitive_or_legal(self, rule_result: RuleResult) -> RuleResult:
        topic = rule_result.detected_topic
        if topic in self.sensitive_topics:
            rule_result.escalation_required = True
            rule_result.escalation_reason = "SENSITIVE_TOPIC"
            rule_result.answer = (
                "This topic is sensitive. I will connect you to an HR specialist for further assistance."
            )
        elif topic in self.legal_topics:
            rule_result.escalation_required = True
            rule_result.escalation_reason = "LEGAL_QUESTION"
            rule_result.answer = (
                "This question may have legal implications. I will connect you to the legal team for further assistance."
            )
        return rule_result

# --- Response Formatter ---

class ResponseFormatter:
    """Formats answers with source citations, disclaimers, and templates for output channels."""
    def format_response(self, rule_result: RuleResult, escalation_ticket_id: Optional[str] = None) -> FormattedResponse:
        if rule_result.escalation_required:
            return FormattedResponse(
                success=False,
                answer=rule_result.answer,
                cited_source=rule_result.cited_source,
                cited_section=rule_result.cited_section,
                cited_last_updated=rule_result.cited_last_updated,
                disclaimer=rule_result.disclaimer,
                escalation_ticket_id=escalation_ticket_id,
                error_code=rule_result.escalation_reason,
                error_message="Escalation required.",
                error_type="ESCALATION_REQUIRED",
                fixing_tips="Your question has been routed to a specialist. You will be contacted soon."
            )
        else:
            return FormattedResponse(
                success=True,
                answer=rule_result.answer,
                cited_source=rule_result.cited_source,
                cited_section=rule_result.cited_section,
                cited_last_updated=rule_result.cited_last_updated,
                disclaimer=rule_result.disclaimer
            )

# --- Escalation Service ---

class EscalationService:
    """Creates and routes escalation tickets to HR or Legal, tracks status."""
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def escalate(self, query: UserQuery, reason: str) -> Optional[str]:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "user_id": query.user_id,
            "query": query.query,
            "reason": reason
        }
        try:
            _obs_t0 = _time.time()
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=5)
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            if resp.status_code == 200:
                data = resp.json()
                return data.get("ticket_id")
            else:
                logger.error(f"Escalation API error: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Escalation API exception: {e}")
            return None

# --- Unanswered Question Logger ---

class UnansweredQuestionLogger:
    """Logs unanswered or ambiguous questions for HR review and analytics."""
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def log_unanswered(self, query: UserQuery, context: ProcessedQuery):
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "user_id": query.user_id,
            "query": query.query,
            "sanitized_query": context.sanitized_query,
            "session_context": context.session_context
        }
        try:
            _obs_t0 = _time.time()
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=3)
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            if resp.status_code != 200:
                logger.warning(f"Logging API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Logging API exception: {e}")

# --- HR Admin Panel Adapter ---

class HRAdminPanelAdapter:
    """Handles policy document uploads, invalidations, and version tracking."""
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def update_knowledge_base(self, document: Dict[str, Any]) -> bool:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            _obs_t0 = _time.time()
            resp = requests.post(self.api_url, headers=headers, json=document, timeout=5)
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"HR Admin API exception: {e}")
            return False

# --- Main Agent Class ---

class BaseAgent:
    pass

class CompanyPolicyQAAgent(BaseAgent):
    """
    Main agent class that composes all supporting services and orchestrates the Q&A workflow.
    """
    def __init__(self):
        # Compose all services
        self.pii_redactor = PIIRedactor()
        self.session_manager = SessionManager(Config.REDIS_URL, Config.SESSION_TTL_SECONDS)
        self.query_processor = QueryProcessor(self.session_manager, self.pii_redactor)
        self.azure_search = AzureAISearchConnector(
            Config.AZURE_AI_SEARCH_ENDPOINT,
            Config.AZURE_AI_SEARCH_KEY,
            Config.AZURE_AI_SEARCH_INDEX
        )
        self.llm_orchestrator = LLMOrchestrator(
            Config.OPENAI_API_KEY,
            {
                "model": "gpt-4.1",
                "temperature": 0.7,
                "max_tokens": 2000,
                "system_prompt": (
                    "You are the Company Policy Q&A Agent. Your role is to answer employee questions about company policies, HR procedures, and handbook content using only the official, indexed documents. Every answer must cite the source document, section, and last-updated date. If you are not confident in your answer or the topic is sensitive or legal, escalate to the appropriate specialist. Never fabricate details or provide legal advice. Keep answers concise and professional, and offer more detail if requested."
                ),
                "few_shot_examples": [
                    "Q: How many sick days do I get per year? A: According to Section 3.1 of the Leave Policy (updated Jan 2026), full-time employees are entitled to 10 paid sick days per year.",
                    "Q: Can I work from home on Fridays? A: According to Section 5.2 of the Remote Work Policy (updated Mar 2026), employees may work from home on Fridays with manager approval.",
                    "Q: What happens to my unvested stock if I resign? A: According to Section 7.4 of the Equity Policy (updated Feb 2026), unvested stock options are forfeited upon resignation. For further details or legal implications, I will connect you to HR/Legal.",
                    "Q: What is the process for requesting parental leave? A: According to Section 4.3 of the Leave Policy (updated Jan 2026), to request parental leave, submit the Parental Leave Request Form to HR at least 30 days in advance. Would you like a link to the form?"
                ]
            }
        )
        self.business_rule_engine = BusinessRuleEngine()
        self.response_formatter = ResponseFormatter()
        self.escalation_service = EscalationService(Config.ESCALATION_API_URL, Config.ESCALATION_API_KEY)
        self.unanswered_logger = UnansweredQuestionLogger(Config.LOGGING_API_URL, Config.LOGGING_API_KEY)
        self.hr_admin_adapter = HRAdminPanelAdapter(Config.HR_ADMIN_API_URL, Config.HR_ADMIN_API_KEY)

    @trace_agent(agent_name='Company Policy Q&A Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_query(self, user_query: UserQuery) -> FormattedResponse:
        try:
            # Step 1: Process and sanitize query
            processed_query = self.query_processor.process_query(user_query)
            # Step 2: Retrieve relevant policy chunks
            policy_chunks = self.azure_search.search_documents(processed_query)
            if not policy_chunks:
                await self.unanswered_logger.log_unanswered(user_query, processed_query)
                return FormattedResponse(
                    success=False,
                    error_code="NO_ANSWER_FOUND",
                    error_message="No relevant policy information found.",
                    error_type="NO_ANSWER_FOUND",
                    fixing_tips="Try rephrasing your question or contact HR for assistance."
                )
            # Step 3: Construct prompt and call LLM
            llm_input = LLMInput(
                system_prompt=self.llm_orchestrator.system_prompt,
                user_prompt=processed_query.sanitized_query,
                context_chunks=policy_chunks,
                few_shot_examples=self.llm_orchestrator.few_shot_examples,
                session_context=processed_query.session_context
            )
            llm_output = await self.llm_orchestrator.generate_answer(llm_input)
            # Step 4: Apply business rules
            rule_result = self.business_rule_engine.apply_rules(llm_output, processed_query)
            rule_result = self.business_rule_engine.check_sensitive_or_legal(rule_result)
            # Step 5: Escalation if required
            escalation_ticket_id = None
            if rule_result.escalation_required:
                escalation_ticket_id = await self.escalation_service.escalate(user_query, rule_result.escalation_reason or "ESCALATION_REQUIRED")
                await self.unanswered_logger.log_unanswered(user_query, processed_query)
            # Step 6: Format response
            response = self.response_formatter.format_response(rule_result, escalation_ticket_id)
            # Step 7: Update session context
            self.session_manager.update_session(user_query.user_id, {
                "last_query": user_query.query,
                "last_answer": response.answer,
                "last_cited_source": response.cited_source,
                "last_cited_section": response.cited_section,
                "last_cited_last_updated": response.cited_last_updated
            })
            return response
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return FormattedResponse(
                success=False,
                error_code="INTERNAL_ERROR",
                error_message=str(e),
                error_type="INTERNAL_ERROR",
                fixing_tips="Please try again later or contact support."
            )

# --- FastAPI App and Endpoints ---

app = FastAPI(
    title="Company Policy Q&A Agent",
    description="Answers employee questions about company policies, HR procedures, and handbook content.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CompanyPolicyQAAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "INVALID_INPUT",
            "error_message": "Input validation failed.",
            "error_type": "ValidationError",
            "fixing_tips": "Check your input for missing fields, excessive length, or invalid characters. Ensure JSON is well-formed."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": "HTTP_ERROR",
            "error_message": exc.detail,
            "error_type": "HTTPException",
            "fixing_tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "error_message": str(exc),
            "error_type": "InternalServerError",
            "fixing_tips": "Please try again later or contact support."
        }
    )

@app.post("/ask", response_model=FormattedResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def ask_policy_question(query: UserQuery):
    """
    Endpoint to ask a company policy or HR question.
    """
    try:
        response = await agent.answer_query(query)
        return response
    except ValidationError as ve:
        logger.warning(f"Validation error in /ask: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error_code": "INVALID_INPUT",
                "error_message": str(ve),
                "error_type": "ValidationError",
                "fixing_tips": "Check your input for missing fields, excessive length, or invalid characters. Ensure JSON is well-formed."
            }
        )
    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_message": str(e),
                "error_type": "InternalServerError",
                "fixing_tips": "Please try again later or contact support."
            }
        )

@app.post("/admin/update_kb")
async def update_knowledge_base(document: Dict[str, Any]):
    """
    Endpoint for HR admins to update the knowledge base.
    """
    try:
        success = await agent.hr_admin_adapter.update_knowledge_base(document)
        if success:
            return {"success": True, "message": "Knowledge base updated."}
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "KB_UPDATE_FAILED",
                    "error_message": "Failed to update knowledge base.",
                    "error_type": "UpdateError",
                    "fixing_tips": "Check document format and try again."
                }
            )
    except Exception as e:
        logger.error(f"Error in /admin/update_kb: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_message": str(e),
                "error_type": "InternalServerError",
                "fixing_tips": "Please try again later or contact support."
            }
        )

@app.middleware("http")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def catch_json_errors(request: Request, call_next):
    """
    Middleware to catch malformed JSON and provide helpful error messages.
    """
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            await request.json()
        except Exception as e:
            logger.warning(f"Malformed JSON: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_code": "MALFORMED_JSON",
                    "error_message": "Malformed JSON in request body.",
                    "error_type": "JSONDecodeError",
                    "fixing_tips": "Ensure your JSON is properly formatted: use double quotes, check commas, and avoid trailing commas."
                }
            )
    response = await call_next(request)
    return response

# --- Main Execution Block ---



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
        logger.info("Starting Company Policy Q&A Agent API server...")
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