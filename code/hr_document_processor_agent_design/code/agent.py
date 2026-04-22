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
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, ValidationError, validator, root_validator
from dotenv import load_dotenv
import openai
import spacy
import pdfplumber
from docx import Document as DocxDocument
from PIL import Image
import numpy as np
import tesserocr
import requests

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HRDocumentProcessorAgent")

# OAuth2 setup (dummy for demo; replace with real SSO/OAuth2 in production)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration management
class Config(BaseModel):
    OPENAI_API_KEY: str = Field(..., description="OpenAI API Key")
    AZURE_AI_SEARCH_ENDPOINT: str = Field(..., description="Azure AI Search Endpoint")
    AZURE_AI_SEARCH_KEY: str = Field(..., description="Azure AI Search API Key")
    ROLE_ACCESS_CONTROL: Dict[str, List[str]] = Field(..., description="Role access control mapping")
    MAX_TEXT_SIZE: int = Field(50000, description="Maximum allowed text size")
    SUPPORTED_FORMATS: List[str] = Field(
        ["pdf", "docx", "png", "jpg", "jpeg"],
        description="Supported document formats"
    )
    AUDIT_LOG_RETENTION_DAYS: int = Field(90, description="Audit log retention period")
    LLM_MODEL: str = Field("gpt-4.1", description="LLM model name")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(2000, description="LLM max tokens")

    @root_validator
    def validate_config(cls, values):
        missing = []
        for key in ["OPENAI_API_KEY", "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_KEY", "ROLE_ACCESS_CONTROL"]:
            if not values.get(key):
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        return values

def get_config() -> Config:
    try:
        config = Config(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            AZURE_AI_SEARCH_ENDPOINT=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            AZURE_AI_SEARCH_KEY=os.getenv("AZURE_AI_SEARCH_KEY"),
            ROLE_ACCESS_CONTROL={
                "user": ["upload"],
                "hr_manager": ["upload", "batch_process"],
                "admin": ["upload", "batch_process", "audit"]
            }
        )
        return config
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise

config = get_config()

# Initialize OpenAI client (async)
openai_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# Load spaCy NER model
try:
    _obs_t0 = _time.time()
    nlp = spacy.load("en_core_web_sm")
    try:
        trace_tool_call(
            tool_name='spacy.load',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(nlp)[:200] if nlp is not None else None,
            status="success",
        )
    except Exception:
        pass
except Exception:
    nlp = spacy.blank("en")

# FastAPI app setup
app = FastAPI(
    title="HR Document Processor Agent",
    description="Automated HR document processing with LLM, OCR, NER, PII masking, and audit logging.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class UploadRequest(BaseModel):
    user_role: str = Field(..., description="User role")
    employee_id: Optional[str] = Field(None, description="Employee ID")
    document_type: Optional[str] = Field(None, description="Document type")
    document_text: Optional[str] = Field(None, description="Extracted document text")

    @validator("user_role")
    def validate_role(cls, v):
        if v not in config.ROLE_ACCESS_CONTROL:
            raise ValueError("Invalid user role")
        return v

    @validator("document_text")
    def validate_text(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Document text cannot be empty")
            if len(v) > config.MAX_TEXT_SIZE:
                raise ValueError(f"Document text exceeds {config.MAX_TEXT_SIZE} characters")
        return v

class BatchUploadRequest(BaseModel):
    user_role: str = Field(..., description="User role")
    documents: List[UploadRequest] = Field(..., description="List of documents for batch processing")

    @validator("user_role")
    def validate_role(cls, v):
        if v not in config.ROLE_ACCESS_CONTROL:
            raise ValueError("Invalid user role")
        return v

class AzureSearchRequest(BaseModel):
    query: str = Field(..., description="HR query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

    @validator("query")
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_query(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    error_code: str
    message: str
    tips: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    document_type: Optional[str]
    confidence: Optional[float]
    extracted_fields: Optional[Dict[str, Any]]
    flagged_fields: Optional[List[str]]
    summary_json: Optional[Dict[str, Any]]
    summary_text: Optional[str]
    errors: Optional[List[Dict[str, Any]]] = None

class BatchResponse(BaseModel):
    success: bool
    results: List[UploadResponse]
    errors: Optional[List[Dict[str, Any]]] = None

class AzureSearchResponse(BaseModel):
    success: bool
    results: Any
    errors: Optional[List[Dict[str, Any]]] = None

# --- Utility Functions ---

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_ssn(ssn: str) -> str:
    if not ssn or not isinstance(ssn, str):
        return ""
    return "***-**-" + ssn[-4:] if len(ssn) >= 4 else "***-**-****"

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_tax_id(tax_id: str) -> str:
    if not tax_id or not isinstance(tax_id, str):
        return "****"
    return "****" + tax_id[-4:] if len(tax_id) >= 4 else "****"

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_bank_account(account: str) -> str:
    if not account or not isinstance(account, str):
        return "****"
    return "****" + account[-4:] if len(account) >= 4 else "****"

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_field(field: str, field_type: str) -> str:
    if field_type == "ssn":
        return mask_ssn(field)
    elif field_type == "tax_id":
        return mask_tax_id(field)
    elif field_type == "bank_account":
        return mask_bank_account(field)
    else:
        return field

@with_content_safety(config=GUARDRAILS_CONFIG)
def hash_document(content: str) -> str:
    _obs_t0 = _time.time()
    _obs_resp = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        trace_tool_call(
            tool_name='content.encode',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(_obs_resp)[:200] if _obs_resp is not None else None,
            status="success",
        )
    except Exception:
        pass
    return _obs_resp

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    return text.strip().replace("\r", "").replace("\n", " ").replace('"', "'")

@with_content_safety(config=GUARDRAILS_CONFIG)
def extract_text_from_pdf(file_path: str) -> str:
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return sanitize_text(text)
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return ""

@with_content_safety(config=GUARDRAILS_CONFIG)
def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return sanitize_text(text)
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        return ""

@with_content_safety(config=GUARDRAILS_CONFIG)
def extract_text_from_image(file_path: str) -> str:
    try:
        image = Image.open(file_path)
        text = tesserocr.image_to_text(image)
        return sanitize_text(text)
    except Exception as e:
        logger.error(f"Image OCR error: {str(e)}")
        return ""

@with_content_safety(config=GUARDRAILS_CONFIG)
def detect_language(text: str) -> str:
    doc = nlp(text)
    return doc.lang_ if hasattr(doc, "lang_") else "en"

@with_content_safety(config=GUARDRAILS_CONFIG)
def sentiment_analysis(text: str) -> str:
    # Use LLM for sentiment if needed, fallback to neutral
    return "neutral"

@with_content_safety(config=GUARDRAILS_CONFIG)
def entity_extraction(text: str) -> Dict[str, Any]:
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities

@with_content_safety(config=GUARDRAILS_CONFIG)
def text_classification(text: str) -> str:
    # Use LLM for classification
    return "HR Document"

# --- Exception Handlers ---

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_type="ValidationError",
            error_code="ERR_INVALID_INPUT",
            message="Input validation failed. Please check your JSON formatting, quotes, commas, and content.",
            tips="Ensure all required fields are present and formatted correctly. Avoid special characters in field names."
        ).dict()
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type="InternalError",
            error_code="ERR_INTERNAL_ERROR",
            message="An unexpected error occurred. Please try again or contact support.",
            tips="Check your input and retry. If the issue persists, contact IT."
        ).dict()
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_type="HTTPException",
            error_code="ERR_HTTP_ERROR",
            message=exc.detail,
            tips="Check your request and authentication."
        ).dict()
    )

# --- Authentication Service ---

class AuthenticationService:
    """Handles OAuth2 authentication and role-based access control."""

    @staticmethod
    async def authenticate(token: str) -> str:
        # Dummy implementation; replace with real SSO/OAuth2
        if token == "hr_manager_token":
            return "hr_manager"
        elif token == "admin_token":
            return "admin"
        else:
            return "user"

    @staticmethod
    def authorize(role: str, action: str) -> bool:
        return action in config.ROLE_ACCESS_CONTROL.get(role, [])

# --- Audit Logging Service ---

class AuditLoggingService:
    """Logs all document access, extraction, and compliance events."""

    @staticmethod
    def log_event(event_type: str, metadata: Dict[str, Any]):
        masked_metadata = PIIMaskingService.mask_in_logs(metadata)
        logger.info(f"AuditLog [{event_type}]: {masked_metadata}")

# --- PII Masking Service ---

class PIIMaskingService:
    """Masks sensitive PII in outputs and logs."""

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def mask_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
        masked = {}
        for k, v in fields.items():
            if k.lower() in ["ssn", "tax_id", "bank_account"]:
                masked[k] = mask_field(v, k.lower())
            else:
                masked[k] = v
        return masked

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def mask_in_logs(log_entry: Dict[str, Any]) -> Dict[str, Any]:
        return PIIMaskingService.mask_fields(log_entry)

# --- Duplicate Detection Service ---

class DuplicateDetectionService:
    """Detects duplicate document uploads."""

    _metadata_cache = {}

    @classmethod
    def check_duplicate(cls, employee_id: str, document_hash: str) -> bool:
        if not employee_id or not document_hash:
            return False
        key = f"{employee_id}:{document_hash}"
        if key in cls._metadata_cache:
            AuditLoggingService.log_event("duplicate_detected", {"employee_id": employee_id, "document_hash": document_hash})
            return True
        cls._metadata_cache[key] = True
        return False

# --- Completeness Checker ---

class CompletenessChecker:
    """Checks for presence of all required onboarding documents and fields."""

    REQUIRED_DOCS = [
        "offer_letter",
        "W-4",
        "I-9",
        "NDA",
        "employment_contract",
        "personal_info_form"
    ]

    @staticmethod
    def check_required_documents(uploaded_documents: List[str]) -> List[str]:
        missing = [doc for doc in CompletenessChecker.REQUIRED_DOCS if doc not in uploaded_documents]
        return missing

    @staticmethod
    def list_missing_fields(fields: Dict[str, Any]) -> List[str]:
        required_fields = [
            "full_name", "employee_id", "start_date", "role", "department",
            "location", "salary_band", "tax_withholding_info", "emergency_contact"
        ]
        missing = [f for f in required_fields if f not in fields or not fields.get(f)]
        return missing

# --- Document Classification Engine ---

class DocumentClassificationEngine:
    """Classifies document type using LLM."""

    @staticmethod
    async def classify(document_text: str) -> Tuple[str, float]:
        try:
            _obs_t0 = _time.time()
            response = await openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": config.llm_configuration["system_prompt"]},
                    {"role": "user", "content": f"Classify this HR document: {document_text[:1000]}"}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=200
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
            content = response.choices[0].message.content
            # Parse LLM response for document type and confidence
            doc_type = "Unknown"
            confidence = 0.0
            if "Document type:" in content:
                doc_type = content.split("Document type:")[1].split("(")[0].strip()
                if "confidence:" in content:
                    conf_str = content.split("confidence:")[1].split(")")[0].strip()
                    try:
                        confidence = float(conf_str)
                    except Exception:
                        confidence = 0.0
            return doc_type, confidence
        except Exception as e:
            logger.error(f"LLM classification error: {str(e)}")
            return "Unknown", 0.0

# --- Field Extraction Engine ---

class FieldExtractionEngine:
    """Extracts required HR fields using OCR and NER."""

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_fields(document_type: str, document_content: str) -> Dict[str, Any]:
        try:
            # Use LLM for NER and field extraction
            _obs_t0 = _time.time()
            response = await openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": config.llm_configuration["system_prompt"]},
                    {"role": "user", "content": f"Extract HR fields from this {document_type} document: {document_content[:3000]}"}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=500
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
            content = response.choices[0].message.content
            # Attempt to parse JSON from LLM output
            try:
                import json
                fields = json.loads(content)
            except Exception:
                # Fallback: use spaCy NER
                fields = entity_extraction(document_content)
            return fields
        except Exception as e:
            logger.error(f"LLM extraction error: {str(e)}")
            return {}

    @staticmethod
    def validate_fields(fields: Dict[str, Any]) -> List[str]:
        return CompletenessChecker.list_missing_fields(fields)

    @staticmethod
    async def score_confidence(fields: Dict[str, Any]) -> Dict[str, float]:
        # Use LLM for confidence scoring
        try:
            _obs_t0 = _time.time()
            response = await openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": config.llm_configuration["system_prompt"]},
                    {"role": "user", "content": f"Score confidence for these HR fields: {str(fields)}"}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=200
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
            content = response.choices[0].message.content
            import json
            try:
                scores = json.loads(content)
            except Exception:
                scores = {k: 0.8 for k in fields.keys()}
            return scores
        except Exception as e:
            logger.error(f"LLM confidence scoring error: {str(e)}")
            return {k: 0.8 for k in fields.keys()}

# --- Summary Generation Service ---

class SummaryGenerationService:
    """Generates structured JSON and human-readable summaries."""

    @staticmethod
    async def generate_json_summary(fields: Dict[str, Any]) -> Dict[str, Any]:
        masked_fields = PIIMaskingService.mask_fields(fields)
        return masked_fields

    @staticmethod
    async def generate_human_summary(fields: Dict[str, Any]) -> str:
        masked_fields = PIIMaskingService.mask_fields(fields)
        summary = "Onboarding Summary:\n"
        for k, v in masked_fields.items():
            summary += f"{k}: {v}\n"
        return summary

# --- Azure AI Search Client ---

class AzureAISearchClient:
    """Retrieves HR policy and document standards."""

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def query_knowledge_base(query: str, filters: Optional[Dict[str, Any]] = None) -> Any:
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": config.AZURE_AI_SEARCH_KEY
            }
            payload = {"query": query}
            if filters:
                payload["filters"] = filters
            _obs_t0 = _time.time()
            response = requests.post(
                config.AZURE_AI_SEARCH_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=5
            )
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(response)[:200] if response is not None else None,
                    status="success",
                )
            except Exception:
                pass
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Azure AI Search error: {response.text}")
                return {"error": response.text}
        except Exception as e:
            logger.error(f"Azure AI Search exception: {str(e)}")
            return {"error": str(e)}

# --- Document Ingestion Service ---

class DocumentIngestionService:
    """Accepts and validates document uploads."""

    @staticmethod
    def validate_format(filename: str) -> bool:
        ext = filename.split(".")[-1].lower()
        return ext in config.SUPPORTED_FORMATS

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def trigger_processing(document_text: str) -> str:
        return sanitize_text(document_text)

# --- Batch Processing Controller ---

class BatchProcessingController:
    """Allows HR managers to process multiple documents in batch."""

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_batch(documents: List[UploadRequest], user_role: str) -> List[UploadResponse]:
        results = []
        for doc in documents:
            try:
                agent = HRDocumentProcessorAgent()
                resp = await agent.process_document(doc)
                results.append(resp)
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                results.append(UploadResponse(
                    success=False,
                    errors=[{
                        "error_type": "BatchProcessingError",
                        "error_code": "ERR_BATCH_PROCESS",
                        "message": str(e),
                        "tips": "Check document format and content."
                    }]
                ))
        return results

# --- Main Agent Class ---

class BaseAgent:
    """Base class for agent."""

class HRDocumentProcessorAgent(BaseAgent):
    """HR Document Processor Agent."""

    def __init__(self):
        self.ingestion_service = DocumentIngestionService()
        self.classification_engine = DocumentClassificationEngine()
        self.field_extraction_engine = FieldExtractionEngine()
        self.completeness_checker = CompletenessChecker()
        self.duplicate_detection_service = DuplicateDetectionService()
        self.pii_masking_service = PIIMaskingService()
        self.audit_logging_service = AuditLoggingService()
        self.summary_generation_service = SummaryGenerationService()
        self.batch_processing_controller = BatchProcessingController()
        self.azure_ai_search_client = AzureAISearchClient()
        self.authentication_service = AuthenticationService()

    @trace_agent(agent_name='HR Document Processor Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_document(self, upload_req: UploadRequest) -> UploadResponse:
        """Process a single HR document."""
        try:
            # Validate role
            if not self.authentication_service.authorize(upload_req.user_role, "upload"):
                return UploadResponse(
                    success=False,
                    errors=[{
                        "error_type": "AccessDenied",
                        "error_code": "ERR_ACCESS_DENIED",
                        "message": "User role not authorized for upload.",
                        "tips": "Contact HR for access."
                    }]
                )

            # Validate document text
            if not upload_req.document_text or len(upload_req.document_text) > config.MAX_TEXT_SIZE:
                return UploadResponse(
                    success=False,
                    errors=[{
                        "error_type": "InvalidDocument",
                        "error_code": "ERR_INVALID_DOCUMENT",
                        "message": "Document text missing or too large.",
                        "tips": f"Ensure document text is present and <= {config.MAX_TEXT_SIZE} characters."
                    }]
                )

            # Duplicate detection
            doc_hash = hash_document(upload_req.document_text)
            if upload_req.employee_id and self.duplicate_detection_service.check_duplicate(upload_req.employee_id, doc_hash):
                return UploadResponse(
                    success=False,
                    errors=[{
                        "error_type": "DuplicateDocument",
                        "error_code": "ERR_DUPLICATE_DOCUMENT",
                        "message": "Duplicate document detected.",
                        "tips": "Do not upload the same document twice."
                    }]
                )

            # Document classification
            doc_type, confidence = await self.classification_engine.classify(upload_req.document_text)
            if confidence < 0.85:
                self.audit_logging_service.log_event("low_confidence_classification", {
                    "employee_id": upload_req.employee_id,
                    "document_type": doc_type,
                    "confidence": confidence
                })
                return UploadResponse(
                    success=False,
                    document_type=doc_type,
                    confidence=confidence,
                    errors=[{
                        "error_type": "LowConfidence",
                        "error_code": "ERR_LOW_CONFIDENCE",
                        "message": f"Document classification confidence too low ({confidence}).",
                        "tips": "Upload a clearer document or contact HR."
                    }]
                )

            # Field extraction
            fields = await self.field_extraction_engine.extract_fields(doc_type, upload_req.document_text)
            missing_fields = self.field_extraction_engine.validate_fields(fields)
            confidence_scores = await self.field_extraction_engine.score_confidence(fields)
            flagged_fields = [f for f, score in confidence_scores.items() if score < 0.65]

            # Completeness check
            if missing_fields:
                self.audit_logging_service.log_event("missing_fields", {
                    "employee_id": upload_req.employee_id,
                    "missing_fields": missing_fields
                })

            # Mask PII
            masked_fields = self.pii_masking_service.mask_fields(fields)

            # Generate summaries
            summary_json = await self.summary_generation_service.generate_json_summary(masked_fields)
            summary_text = await self.summary_generation_service.generate_human_summary(masked_fields)

            # Audit log
            self.audit_logging_service.log_event("document_processed", {
                "employee_id": upload_req.employee_id,
                "document_type": doc_type,
                "confidence": confidence,
                "flagged_fields": flagged_fields,
                "missing_fields": missing_fields
            })

            return UploadResponse(
                success=True,
                document_type=doc_type,
                confidence=confidence,
                extracted_fields=masked_fields,
                flagged_fields=flagged_fields,
                summary_json=summary_json,
                summary_text=summary_text,
                errors=[{
                    "error_type": "MissingFields",
                    "error_code": "ERR_MISSING_FIELD",
                    "message": f"Missing fields: {missing_fields}" if missing_fields else "",
                    "tips": "Provide all required HR fields."
                }] if missing_fields else None
            )
        except Exception as e:
            self.audit_logging_service.log_event("processing_error", {
                "employee_id": upload_req.employee_id,
                "error": str(e)
            })
            return UploadResponse(
                success=False,
                errors=[{
                    "error_type": "ProcessingError",
                    "error_code": "ERR_PROCESSING",
                    "message": str(e),
                    "tips": "Check document format and content."
                }]
            )

    @trace_agent(agent_name='HR Document Processor Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_batch(self, batch_req: BatchUploadRequest) -> BatchResponse:
        """Process batch of HR documents."""
        if not self.authentication_service.authorize(batch_req.user_role, "batch_process"):
            return BatchResponse(
                success=False,
                results=[],
                errors=[{
                    "error_type": "AccessDenied",
                    "error_code": "ERR_ACCESS_DENIED",
                    "message": "User role not authorized for batch processing.",
                    "tips": "Contact HR for access."
                }]
            )
        results = await self.batch_processing_controller.process_batch(batch_req.documents, batch_req.user_role)
        return BatchResponse(success=True, results=results)

    @trace_agent(agent_name='HR Document Processor Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def query_azure_ai_search(self, search_req: AzureSearchRequest) -> AzureSearchResponse:
        """Query Azure AI Search for HR knowledge base."""
        results = await self.azure_ai_search_client.query_knowledge_base(search_req.query, search_req.filters)
        return AzureSearchResponse(success=True, results=results)

# --- FastAPI Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_role: str = Form(...),
    employee_id: str = Form(None)
):
    """Upload and process a single HR document."""
    try:
        filename = file.filename
        if not DocumentIngestionService.validate_format(filename):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_type="UnsupportedFormat",
                    error_code="ERR_UNSUPPORTED_FORMAT",
                    message=f"Unsupported file format: {filename}",
                    tips=f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
                ).dict()
            )
        # Save file temporarily in memory
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True, suffix=f".{filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            tmp.flush()
            ext = filename.split(".")[-1].lower()
            if ext == "pdf":
                document_text = extract_text_from_pdf(tmp.name)
            elif ext == "docx":
                document_text = extract_text_from_docx(tmp.name)
            elif ext in ["png", "jpg", "jpeg"]:
                document_text = extract_text_from_image(tmp.name)
            else:
                document_text = ""
        if not document_text:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_type="ExtractionError",
                    error_code="ERR_EXTRACTION_FAILED",
                    message="Failed to extract text from document.",
                    tips="Upload a clearer document or try a different format."
                ).dict()
            )
        upload_req = UploadRequest(
            user_role=user_role,
            employee_id=employee_id,
            document_text=document_text
        )
        agent = HRDocumentProcessorAgent()
        response = await agent.process_document(upload_req)
        return JSONResponse(status_code=200, content=response.dict())
    except ValidationError as ve:
        logger.error(f"Upload validation error: {ve.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_type="ValidationError",
                error_code="ERR_INVALID_INPUT",
                message="Input validation failed. Please check your JSON formatting, quotes, commas, and content.",
                tips="Ensure all required fields are present and formatted correctly. Avoid special characters in field names."
            ).dict()
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_type="InternalError",
                error_code="ERR_INTERNAL_ERROR",
                message="An unexpected error occurred during upload.",
                tips="Check your input and retry. If the issue persists, contact IT."
            ).dict()
        )

@app.post("/batch", response_model=BatchResponse)
async def batch_upload(batch_req: BatchUploadRequest):
    """Batch upload and process HR documents (HR manager only)."""
    try:
        agent = HRDocumentProcessorAgent()
        response = await agent.process_batch(batch_req)
        return JSONResponse(status_code=200, content=response.dict())
    except ValidationError as ve:
        logger.error(f"Batch validation error: {ve.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_type="ValidationError",
                error_code="ERR_INVALID_INPUT",
                message="Input validation failed. Please check your JSON formatting, quotes, commas, and content.",
                tips="Ensure all required fields are present and formatted correctly. Avoid special characters in field names."
            ).dict()
        )
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_type="InternalError",
                error_code="ERR_INTERNAL_ERROR",
                message="An unexpected error occurred during batch upload.",
                tips="Check your input and retry. If the issue persists, contact IT."
            ).dict()
        )

@app.post("/azure-search", response_model=AzureSearchResponse)
async def azure_search(search_req: AzureSearchRequest):
    """Query Azure AI Search for HR knowledge base."""
    try:
        agent = HRDocumentProcessorAgent()
        response = await agent.query_azure_ai_search(search_req)
        return JSONResponse(status_code=200, content=response.dict())
    except ValidationError as ve:
        logger.error(f"Azure search validation error: {ve.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_type="ValidationError",
                error_code="ERR_INVALID_INPUT",
                message="Input validation failed. Please check your JSON formatting, quotes, commas, and content.",
                tips="Ensure all required fields are present and formatted correctly. Avoid special characters in field names."
            ).dict()
        )
    except Exception as e:
        logger.error(f"Azure search error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_type="InternalError",
                error_code="ERR_INTERNAL_ERROR",
                message="An unexpected error occurred during Azure search.",
                tips="Check your input and retry. If the issue persists, contact IT."
            ).dict()
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"success": True, "status": "healthy"}

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
        logger.info("Starting HR Document Processor Agent API...")
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