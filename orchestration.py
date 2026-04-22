import logging
import asyncio
from typing import Any, Dict, Optional, List

import importlib

logger = logging.getLogger(__name__)

# Dynamically import agent modules
company_policy_qa_agent_mod = importlib.import_module("code.company_policy_qa_agent_design.agent")
it_setup_guide_agent_mod = importlib.import_module("code.it_setup_guide_agent_design.agent")
hr_document_processor_agent_mod = importlib.import_module("code.hr_document_processor_agent_design.agent")

# Import Pydantic models from each agent for input validation and construction
CompanyPolicyQAAgent_UserQuery = getattr(company_policy_qa_agent_mod, "UserQuery")
CompanyPolicyQAAgent_FormattedResponse = getattr(company_policy_qa_agent_mod, "FormattedResponse")

ITSetupGuideAgent_UserContext = getattr(it_setup_guide_agent_mod, "UserContext")
ITSetupGuideAgent_StepInput = getattr(it_setup_guide_agent_mod, "StepInput")
ITSetupGuideAgent_TicketRequest = getattr(it_setup_guide_agent_mod, "TicketRequest")
ITSetupGuideAgent_StepResponse = getattr(it_setup_guide_agent_mod, "StepResponse")
ITSetupGuideAgent_ErrorResponse = getattr(it_setup_guide_agent_mod, "ErrorResponse")
ITSetupGuideAgent_TicketResponse = getattr(it_setup_guide_agent_mod, "TicketResponse")
ITSetupGuideAgent_CompletionResponse = getattr(it_setup_guide_agent_mod, "CompletionResponse")

HRDocumentProcessorAgent_UploadRequest = getattr(hr_document_processor_agent_mod, "UploadRequest")
HRDocumentProcessorAgent_BatchUploadRequest = getattr(hr_document_processor_agent_mod, "BatchUploadRequest")
HRDocumentProcessorAgent_AzureSearchRequest = getattr(hr_document_processor_agent_mod, "AzureSearchRequest")
HRDocumentProcessorAgent_UploadResponse = getattr(hr_document_processor_agent_mod, "UploadResponse")
HRDocumentProcessorAgent_BatchResponse = getattr(hr_document_processor_agent_mod, "BatchResponse")
HRDocumentProcessorAgent_AzureSearchResponse = getattr(hr_document_processor_agent_mod, "AzureSearchResponse")

# Entrypoint functions from each agent
company_policy_qa_ask_policy_question = getattr(company_policy_qa_agent_mod, "ask_policy_question")
it_setup_guide_start_onboarding = getattr(it_setup_guide_agent_mod, "start_onboarding")
it_setup_guide_process_step = getattr(it_setup_guide_agent_mod, "process_step")
it_setup_guide_create_ticket = getattr(it_setup_guide_agent_mod, "create_ticket")
hr_document_processor_upload_document = getattr(hr_document_processor_agent_mod, "upload_document")
hr_document_processor_batch_upload = getattr(hr_document_processor_agent_mod, "batch_upload")
hr_document_processor_azure_search = getattr(hr_document_processor_agent_mod, "azure_search")

class OrchestrationEngine:
    """
    Orchestrates the workflow across:
      1. Company Policy Q&A Agent
      2. IT Setup Guide Agent
      3. HR Document Processor Agent

    Usage:
        engine = OrchestrationEngine()
        result = await engine.execute(input_data)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the workflow:
          1. Call Company Policy Q&A Agent with input_data['policy_qa'].
          2. Call IT Setup Guide Agent with input_data['it_setup'].
          3. Call HR Document Processor Agent with input_data['hr_doc'].
        Each step's output is included in the final result.
        Errors are logged and included in the output.
        """
        results = {}
        errors: List[Dict[str, Any]] = []

        # --- Step 1: Company Policy Q&A Agent ---
        try:
            policy_qa_input = input_data.get("policy_qa", {})
            user_query = CompanyPolicyQAAgent_UserQuery(**policy_qa_input)
            policy_qa_response = await company_policy_qa_ask_policy_question(user_query)
            # Convert to dict if it's a Pydantic model
            if hasattr(policy_qa_response, "model_dump"):
                policy_qa_response_dict = policy_qa_response.model_dump()
            elif hasattr(policy_qa_response, "dict"):
                policy_qa_response_dict = policy_qa_response.dict()
            else:
                policy_qa_response_dict = dict(policy_qa_response)
            results["policy_qa"] = policy_qa_response_dict
        except Exception as e:
            self.logger.error(f"Error in Company Policy Q&A Agent: {e}", exc_info=True)
            errors.append({
                "step": "policy_qa",
                "error": str(e)
            })
            results["policy_qa"] = {
                "success": False,
                "error_type": "INTERNAL_ERROR",
                "error_message": str(e)
            }

        # --- Step 2: IT Setup Guide Agent ---
        try:
            it_setup_input = input_data.get("it_setup", {})
            # Determine which endpoint to call based on input
            # Priority: /start if 'start_onboarding' key, /step if 'step_input', /ticket if 'ticket'
            if "start_onboarding" in it_setup_input:
                ctx = ITSetupGuideAgent_UserContext(**it_setup_input["start_onboarding"])
                it_setup_response = await it_setup_guide_start_onboarding(ctx)
            elif "step_input" in it_setup_input:
                step_input = ITSetupGuideAgent_StepInput(**it_setup_input["step_input"])
                it_setup_response = await it_setup_guide_process_step(step_input)
            elif "ticket" in it_setup_input:
                ticket_req = it_setup_input["ticket"]
                ticket_obj = ITSetupGuideAgent_TicketRequest(**ticket_req)
                it_setup_response = await it_setup_guide_create_ticket(
                    ticket_obj.employee_id, ticket_obj.error_details, ticket_obj.consent
                )
            else:
                raise ValueError("No valid IT Setup Guide input provided (expected 'start_onboarding', 'step_input', or 'ticket').")
            # Convert to dict if it's a Pydantic model
            if hasattr(it_setup_response, "model_dump"):
                it_setup_response_dict = it_setup_response.model_dump()
            elif hasattr(it_setup_response, "dict"):
                it_setup_response_dict = it_setup_response.dict()
            else:
                it_setup_response_dict = dict(it_setup_response)
            results["it_setup"] = it_setup_response_dict
        except Exception as e:
            self.logger.error(f"Error in IT Setup Guide Agent: {e}", exc_info=True)
            errors.append({
                "step": "it_setup",
                "error": str(e)
            })
            results["it_setup"] = {
                "success": False,
                "error_type": "INTERNAL_ERROR",
                "error_message": str(e)
            }

        # --- Step 3: HR Document Processor Agent ---
        try:
            hr_doc_input = input_data.get("hr_doc", {})
            # Determine which endpoint to call based on input
            # Priority: /upload if 'upload', /batch if 'batch', /azure_search if 'azure_search'
            if "upload" in hr_doc_input:
                upload_req = HRDocumentProcessorAgent_UploadRequest(**hr_doc_input["upload"])
                # The upload_document expects file and form fields in FastAPI, but here we assume document_text is already extracted.
                # We'll call the process_document method directly for orchestration.
                agent = hr_document_processor_agent_mod.HRDocumentProcessorAgent()
                upload_response = await agent.process_document(upload_req)
                if hasattr(upload_response, "model_dump"):
                    upload_response_dict = upload_response.model_dump()
                elif hasattr(upload_response, "dict"):
                    upload_response_dict = upload_response.dict()
                else:
                    upload_response_dict = dict(upload_response)
                results["hr_doc"] = upload_response_dict
            elif "batch" in hr_doc_input:
                batch_req = HRDocumentProcessorAgent_BatchUploadRequest(**hr_doc_input["batch"])
                agent = hr_document_processor_agent_mod.HRDocumentProcessorAgent()
                batch_response = await agent.process_batch(batch_req)
                if hasattr(batch_response, "model_dump"):
                    batch_response_dict = batch_response.model_dump()
                elif hasattr(batch_response, "dict"):
                    batch_response_dict = batch_response.dict()
                else:
                    batch_response_dict = dict(batch_response)
                results["hr_doc"] = batch_response_dict
            elif "azure_search" in hr_doc_input:
                azure_req = HRDocumentProcessorAgent_AzureSearchRequest(**hr_doc_input["azure_search"])
                agent = hr_document_processor_agent_mod.HRDocumentProcessorAgent()
                azure_response = await agent.query_azure_ai_search(azure_req)
                if hasattr(azure_response, "model_dump"):
                    azure_response_dict = azure_response.model_dump()
                elif hasattr(azure_response, "dict"):
                    azure_response_dict = azure_response.dict()
                else:
                    azure_response_dict = dict(azure_response)
                results["hr_doc"] = azure_response_dict
            else:
                raise ValueError("No valid HR Document Processor input provided (expected 'upload', 'batch', or 'azure_search').")
        except Exception as e:
            self.logger.error(f"Error in HR Document Processor Agent: {e}", exc_info=True)
            errors.append({
                "step": "hr_doc",
                "error": str(e)
            })
            results["hr_doc"] = {
                "success": False,
                "error_type": "INTERNAL_ERROR",
                "error_message": str(e)
            }

        results["errors"] = errors
        return results

# Convenience function for orchestration
async def run_orchestration(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrypoint for orchestration workflow.
    Args:
        input_data: dict with keys:
            - policy_qa: dict for Company Policy Q&A Agent (user_id, query)
            - it_setup: dict for IT Setup Guide Agent (see docs)
            - hr_doc: dict for HR Document Processor Agent (see docs)
    Returns:
        dict with results from each agent and any errors.
    """
    engine = OrchestrationEngine()
    return await engine.execute(input_data)