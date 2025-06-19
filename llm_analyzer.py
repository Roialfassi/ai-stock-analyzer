# llm_analyzer.py - Advanced LLM Analysis Engine

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import re
import requests

from models import (
    StockData, AnalysisResult, TechnicalIndicators, FinancialMetrics,
    NewsItem, Recommendation, AnalysisType, ChainPromptTemplate,
    SCREENING_CHAIN_TEMPLATE, FUNDAMENTAL_ANALYSIS_CHAIN,
    TECHNICAL_ANALYSIS_CHAIN, NEWS_SENTIMENT_CHAIN
)

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            return response.choices[0].message.content

        except requests.exceptions.RequestException as e: # Order matters, catch more specific exceptions first
            logger.error(f"OpenAI API request failed: {e}")
            return ""
        except ImportError: # Handle if openai library is not found during __init__
             raise # Re-raise to indicate critical failure if provider is used
        except Exception as e: # Catch openai.APIError and other general errors
            logger.error(f"OpenAI API error: {e.__class__.__name__} - {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            logger.warning("No JSON object found in OpenAI response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from OpenAI: {e}. Response snippet: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in OpenAIProvider.parse_json_response: {e.__class__.__name__} - {e}")
            return {}

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        try:
            message = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=temperature,
                system=system_prompt if system_prompt else None,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}")
            return ""
        except ImportError:
            raise
        except Exception as e: # Catch anthropic.APIError and other general errors
            logger.error(f"Anthropic API error: {e.__class__.__name__} - {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            logger.warning("No JSON object found in Anthropic response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Anthropic: {e}. Response snippet: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in AnthropicProvider.parse_json_response: {e.__class__.__name__} - {e}")
            return {}

class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-pro"

    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        try:
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096,
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        data = await response.json()
                        if data.get('candidates') and data['candidates'][0].get('content') and data['candidates'][0]['content'].get('parts'):
                            return data['candidates'][0]['content']['parts'][0]['text']
                        else:
                            logger.error(f"Gemini API error: Unexpected response structure: {await response.text()}")
                            return ""
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        return ""
        except aiohttp.ClientError as e:
            logger.error(f"Gemini API request failed (aiohttp.ClientError): {e}")
            return ""
        except ImportError: # aiohttp not installed
             logger.error("aiohttp is not installed, which is required for GeminiProvider.")
             return "" # Or raise an error
        except Exception as e:
            logger.error(f"Unexpected error in GeminiProvider.complete: {e.__class__.__name__} - {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            logger.warning("No JSON object found in Gemini response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Gemini: {e}. Response snippet: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in GeminiProvider.parse_json_response: {e.__class__.__name__} - {e}")
            return {}

class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider"""

    def __init__(self, api_key: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"

    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Format prompt for instruction-following models
            if system_prompt:
                full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                full_prompt = f"<s>[INST] {prompt} [/INST]"

            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 2048,
                    "return_full_text": False,
                    "do_sample": True,
                    "top_p": 0.95
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        data = await response.json()
                        if data and isinstance(data, list) and data[0].get('generated_text'):
                            return data[0]['generated_text']
                        else:
                            logger.error(f"HuggingFace API error: Unexpected response structure: {await response.text()}")
                            return ""
                    else:
                        error_text = await response.text()
                        logger.error(f"HuggingFace API error: {response.status} - {error_text}")
                        return ""
        except aiohttp.ClientError as e:
            logger.error(f"HuggingFace API request failed (aiohttp.ClientError): {e}")
            return ""
        except ImportError: # aiohttp not installed
             logger.error("aiohttp is not installed, which is required for HuggingFaceProvider.")
             return ""
        except Exception as e:
            logger.error(f"Unexpected error in HuggingFaceProvider.complete: {e.__class__.__name__} - {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            logger.warning("No JSON object found in HuggingFace response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from HuggingFace: {e}. Response snippet: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in HuggingFaceProvider.parse_json_response: {e.__class__.__name__} - {e}")
            return {}

class LMStudioProvider(LLMProvider):
    """LM Studio local server provider"""

    def __init__(self, base_url: str = "http://localhost:1234", model: str = ""):
        self.base_url = base_url.rstrip('/')
        self.model = model  # Optional, LM Studio uses loaded model

    async def complete(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        try:
            url = f"{self.base_url}/v1/chat/completions"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048,
                "stream": False
            }

            # If model is specified, add it to payload
            if self.model:
                payload["model"] = self.model

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        data = await response.json()
                        if data.get('choices') and data['choices'][0].get('message') and data['choices'][0]['message'].get('content'):
                            return data['choices'][0]['message']['content']
                        else:
                            logger.error(f"LM Studio API error: Unexpected response structure: {await response.text()}")
                            return ""
                    else:
                        error_text = await response.text()
                        logger.error(f"LM Studio API error: {response.status} - {error_text}")
                        return ""
        except aiohttp.ClientError as e: # Covers connection errors, timeouts, etc.
            logger.error(f"LM Studio API request failed (aiohttp.ClientError): {e}")
            return ""
        except ImportError: # aiohttp not installed
             logger.error("aiohttp is not installed, which is required for LMStudioProvider.")
             return ""
        except Exception as e:
            logger.error(f"Unexpected error in LMStudioProvider.complete: {e.__class__.__name__} - {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            logger.warning("No JSON object found in LM Studio response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LM Studio: {e}. Response snippet: {response[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in LMStudioProvider.parse_json_response: {e.__class__.__name__} - {e}")
            return {}

# Import aiohttp for async HTTP requests.
# Specific providers will handle ImportErrors if aiohttp is not available when they are initialized/used.
try:
    import aiohttp
except ImportError:
    # This log is a general warning. Each provider that uses aiohttp should also
    # handle its absence more directly, e.g., by raising an error during init or request.
    logger.warning("aiohttp not installed. LLM providers like Gemini, HuggingFace, LMStudio will not work.")
    aiohttp = None # Set to None so checks like 'if aiohttp:' work

class ChainPromptExecutor:
    """Executes multi-step chain prompts"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.context = {}

    async def execute_chain(self, template: ChainPromptTemplate, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of prompts"""
        self.context = initial_context.copy()
        results = {}
        current_step_name = "Unknown"

        try:
            for step_idx, step in enumerate(template.steps):
                current_step_name = step.get('name', f"Step {step_idx + 1}")
                try:
                    # Format prompt with current context
                    prompt = self._format_prompt(step['prompt'], self.context)
                except KeyError as e:
                    logger.error(f"ChainPromptExecutor ({current_step_name}): Missing key {e} in context for prompt templating.")
                    results[step['output']] = f"Error: Missing context variable {e} for step {current_step_name}"
                    # Optionally, decide to halt chain execution here
                    # For now, we'll add the error and let the chain continue if possible,
                    # or subsequent steps might fail due to missing data.
                    self.context[step['output']] = results[step['output']] # Ensure context also reflects the error
                    continue
                except Exception as e:
                    logger.error(f"ChainPromptExecutor ({current_step_name}): Error formatting prompt: {e}", exc_info=True)
                    results[step['output']] = f"Error: Prompt formatting failed for step {current_step_name}"
                    self.context[step['output']] = results[step['output']]
                    continue

                # Execute LLM call
                response = await self.llm.complete(prompt)
                if not response: # Check if LLM call failed (returned empty or None)
                    logger.warning(f"ChainPromptExecutor ({current_step_name}): LLM call returned empty. Prompt snippet: {prompt[:100]}...")
                    results[step['output']] = f"Error: LLM call failed or returned empty for step {current_step_name}."
                    self.context[step['output']] = results[step['output']]
                    # Decide if to continue or not; for now, we continue
                    continue

                # Store result
                output_key = step['output']
                results[output_key] = response
                self.context[output_key] = response # Update context for subsequent steps

                # Parse structured output if needed
                if template.output_format == "json":
                    parsed = self.llm.parse_json_response(response)
                    if parsed: # Successfully parsed
                        self.context[f"{output_key}_parsed"] = parsed
                        results[f"{output_key}_parsed"] = parsed
                    else: # Parsing failed
                        logger.warning(f"ChainPromptExecutor ({current_step_name}): Failed to parse JSON. Response snippet: {response[:200]}...")
                        # Store raw response if parsing fails but LLM call was successful
                        self.context[f"{output_key}_parsed_error"] = "Failed to parse JSON output."
                        results[f"{output_key}_parsed_error"] = self.context[f"{output_key}_parsed_error"]
                        # self.context[f"{output_key}_parsed"] might be left empty or set to a default error dict
                        self.context[f"{output_key}_parsed"] = {"error": "JSON parsing failed"}


        except Exception as e:
            logger.error(f"Unexpected error during chain execution at step {current_step_name}: {e}", exc_info=True)
            results["chain_execution_error"] = f"Error at {current_step_name}: {str(e)}"
        return results

    def _format_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format prompt template with context values"""
        # This basic replacement is prone to errors if keys are missing or if values have braces.
        # A more robust templating engine (e.g., Jinja2) would be better for complex cases.
        # For now, we rely on the KeyError check in execute_chain for missing keys.
        try:
            formatted_template = template
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in formatted_template:
                    if isinstance(value, (dict, list)):
                        try:
                            value_str = json.dumps(value, indent=2)
                        except TypeError as te:
                            logger.warning(f"JSON serialization error for key '{key}' during prompt formatting: {te}. Using str().")
                            value_str = str(value) # Fallback to string representation
                    else:
                        value_str = str(value)
                    formatted_template = formatted_template.replace(placeholder, value_str)

            # Check for any unformatted placeholders (simple check)
            # More sophisticated regex: r"\{[a-zA-Z0-9_]+\}"
            unformatted_placeholders = re.findall(r"\{[^{}\s]+\}", formatted_template)
            if unformatted_placeholders:
                logger.warning(f"Unformatted placeholders remaining in prompt: {unformatted_placeholders}. Prompt snippet: {formatted_template[:200]}...")
            return formatted_template
        except Exception as e:
            logger.error(f"Error formatting prompt template: {e}. Template snippet: {template[:200]}...", exc_info=True)
            raise # Re-raise to be caught by the caller in execute_chain

class StockAnalyzer:
    """Main stock analysis orchestrator"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.chain_executor = ChainPromptExecutor(llm_provider)

    async def analyze_stock(self,
                          stock_data: StockData,
                          financial_metrics: Optional[FinancialMetrics] = None,
                          technical_indicators: Optional[TechnicalIndicators] = None,
                          news_items: Optional[List[NewsItem]] = None,
                          analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE) -> AnalysisResult:
        """Perform comprehensive stock analysis"""

        # Prepare context
        context = {
            'symbol': stock_data.symbol,
            'company_name': stock_data.company_name,
            'stock_data': self._serialize_stock_data(stock_data)
        }

        results = {}
        scores = {}

        # Initialize results and scores with default/error states
        results: Dict[str, Any] = {"overall_analysis_status": "Pending"}
        scores: Dict[str, Optional[float]] = { # Allow None for scores if sub-analysis fails
            'fundamental': None,
            'technical': None,
            'sentiment': None
        }

        try:
            # Run appropriate analysis chains
            if analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.COMPREHENSIVE]:
                if financial_metrics:
                    context['financial_data'] = self._serialize_financial_metrics(financial_metrics)
                    if 'error' in context['financial_data']: # Check for serialization error
                         results['fundamental_error'] = f"Failed to serialize financial data: {context['financial_data']['error']}"
                    else:
                        fundamental_results = await self._run_fundamental_analysis(context)
                        results.update(fundamental_results) # Merge results, may include 'chain_execution_error'
                        if not fundamental_results.get("chain_execution_error"):
                            scores['fundamental'] = await self._score_fundamental_analysis(fundamental_results)
                        else:
                            results['fundamental_error'] = fundamental_results.get("chain_execution_error", "Fundamental analysis chain failed.")
                else:
                    results['fundamental_skipped'] = "Financial metrics not provided."
                    scores['fundamental'] = 0.0 # Or None, depending on how you want to treat skipped analysis


            if analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPREHENSIVE]:
                if technical_indicators:
                    context['technical_indicators'] = self._serialize_technical_indicators(technical_indicators)
                    if 'error' in context['technical_indicators']:
                        results['technical_error'] = f"Failed to serialize technical indicators: {context['technical_indicators']['error']}"
                    else:
                        technical_results = await self._run_technical_analysis(context)
                        results.update(technical_results)
                        if not technical_results.get("chain_execution_error"):
                            scores['technical'] = await self._score_technical_analysis(technical_results)
                        else:
                             results['technical_error'] = technical_results.get("chain_execution_error", "Technical analysis chain failed.")
                else:
                    results['technical_skipped'] = "Technical indicators not provided."
                    scores['technical'] = 0.0

            if analysis_type in [AnalysisType.SENTIMENT, AnalysisType.COMPREHENSIVE]:
                if news_items:
                    serialized_news = [self._serialize_news_item(item) for item in news_items]
                    if any('error' in item for item in serialized_news):
                        results['sentiment_error'] = "Failed to serialize one or more news items."
                        # Optionally provide more detail on which items failed
                    else:
                        context['news_data'] = serialized_news
                        sentiment_results = await self._run_sentiment_analysis(context)
                        results.update(sentiment_results)
                        if not sentiment_results.get("chain_execution_error"):
                            scores['sentiment'] = await self._score_sentiment_analysis(sentiment_results)
                        else:
                            results['sentiment_error'] = sentiment_results.get("chain_execution_error", "Sentiment analysis chain failed.")
                else:
                    results['sentiment_skipped'] = "News items not provided."
                    scores['sentiment'] = 0.0

            results["overall_analysis_status"] = "Completed"
            if any(key.endswith("_error") for key in results):
                results["overall_analysis_status"] = "Completed with errors"
            elif all(key.endswith("_skipped") for key in results if key not in ['symbol', 'company_name', 'stock_data', 'overall_analysis_status']): # Check if all were skipped
                 results["overall_analysis_status"] = "No analysis performed (all data skipped)"


        except Exception as e:
            logger.error(f"Critical error during stock analysis orchestration for {stock_data.symbol}: {e}", exc_info=True)
            results['analysis_orchestration_error'] = str(e)
            # Ensure a default AnalysisResult can be created even if everything fails catastrophically
            return self._create_default_analysis_result(stock_data, f"Orchestration error: {e}", scores, analysis_type)


        # Generate final analysis
        try:
            # Pass the actual analysis_type performed, not always COMPREHENSIVE
            analysis_result = await self._generate_final_analysis(stock_data, results, scores, analysis_type)
        except Exception as e:
            logger.error(f"Error generating final analysis report for {stock_data.symbol}: {e}", exc_info=True)
            analysis_result = self._create_default_analysis_result(stock_data, f"Failed to generate final report: {e}", scores, analysis_type)

        return analysis_result

    def _create_default_analysis_result(self, stock_data: StockData, error_message: str,
                                      scores: Dict[str, Optional[float]],
                                      analysis_type: AnalysisType) -> AnalysisResult:
        """Creates a default AnalysisResult in case of major errors, using provided scores if available."""
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        overall_score_val = sum(valid_scores.values()) / len(valid_scores) if valid_scores else 0.0

        return AnalysisResult(
            stock=stock_data,
            analysis_type=analysis_type, # Reflect the intended analysis type
            llm_summary=f"Analysis could not be fully completed: {error_message}",
            bull_case=["Error in analysis or insufficient data."],
            bear_case=["Error in analysis or insufficient data."],
            fundamental_score=scores.get('fundamental'), # Will be None if not computed
            technical_score=scores.get('technical'),
            sentiment_score=scores.get('sentiment'),
            overall_score=overall_score_val,
            recommendation=self._calculate_recommendation(overall_score_val), # Calculate based on available score
            confidence=0.1, # Very low confidence
            price_target=stock_data.current_price if stock_data and stock_data.current_price is not None else 0.0,
            risk_factors=["Analysis incomplete due to error."],
            catalysts=["Analysis incomplete due to error."]
        )

    async def _run_fundamental_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run fundamental analysis chain with error handling"""
        try:
            # Validate required context keys for this chain if necessary
            if 'financial_data' not in context:
                logger.warning("Fundamental analysis skipped: financial_data missing from context.")
                return {"chain_execution_error": "Missing financial_data for fundamental analysis."}

            # Add peer data (mock for now - consider making this more robust or optional)
            context['peer_list'] = ['AAPL', 'MSFT', 'GOOGL']
            context['peer_data'] = {}
            context['industry_trends'] = "Technology sector showing strong growth"

            results = await self.chain_executor.execute_chain(FUNDAMENTAL_ANALYSIS_CHAIN, context)
            return results
        except Exception as e:
            logger.error(f"Error in _run_fundamental_analysis: {e}", exc_info=True)
            return {"chain_execution_error": f"Fundamental analysis chain failed unexpectedly: {e}"}


    async def _run_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run technical analysis chain with error handling"""
        try:
            if 'technical_indicators' not in context or not isinstance(context['technical_indicators'], dict):
                logger.warning("Technical analysis skipped: technical_indicators missing or invalid in context.")
                return {"chain_execution_error": "Missing or invalid technical_indicators data."}

            # Ensure keys exist before trying to access them
            context['price_data'] = context['technical_indicators'].get('moving_averages', {})
            context['volume_data'] = context['technical_indicators'].get('volume_profile', {})

            results = await self.chain_executor.execute_chain(TECHNICAL_ANALYSIS_CHAIN, context)
            return results
        except Exception as e:
            logger.error(f"Error in _run_technical_analysis: {e}", exc_info=True)
            return {"chain_execution_error": f"Technical analysis chain failed unexpectedly: {e}"}

    async def _run_sentiment_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run news sentiment analysis chain with error handling"""
        try:
            if 'news_data' not in context or not isinstance(context['news_data'], list):
                logger.warning("Sentiment analysis skipped: news_data missing or invalid in context.")
                return {"chain_execution_error": "Missing or invalid news_data for sentiment analysis."}

            results = await self.chain_executor.execute_chain(NEWS_SENTIMENT_CHAIN, context)
            return results
        except Exception as e:
            logger.error(f"Error in _run_sentiment_analysis: {e}", exc_info=True)
            return {"chain_execution_error": f"Sentiment analysis chain failed unexpectedly: {e}"}

    async def _score_analysis(self, analysis_type: str, results: Dict[str, Any], default_score: float = 0.0) -> Optional[float]: # Return Optional[float]
        """Generic scoring function. Returns None if scoring is not possible."""
        # Check for critical errors that would make scoring impossible or meaningless
        if not results or results.get("chain_execution_error") or results.get(f"{analysis_type}_error") or results.get("analysis_orchestration_error"):
            logger.warning(f"Cannot score {analysis_type} due to previous errors or empty/invalid results. Results: {str(results)[:200]}")
            return None # Indicate that scoring was not possible

        prompt_template = """
        Based on the following {analysis_type} analysis, provide a quantitative score from 0-100:
        {json_results}
        
        Consider all relevant factors for {analysis_type} based on the provided data.
        If the data is clearly insufficient, contradictory, or indicates a strong negative/positive outlook that cannot be quantified,
        try to provide a score that reflects this (e.g., very low for negative, 50 for neutral/insufficient).
        
        Respond with ONLY a single number between 0 and 100.
        """
        try:
            # Serialize results carefully, excluding known error keys from the JSON passed to LLM for scoring
            # to avoid confusing the LLM with error messages during the scoring prompt.
            score_input_results = {k: v for k, v in results.items() if not k.endswith('_error') and k != "chain_execution_error"}
            if not score_input_results: # If only errors were present
                logger.warning(f"No valid data to score for {analysis_type} after filtering errors.")
                return None

            json_results = json.dumps(score_input_results, indent=2, default=str)
        except TypeError as e:
            logger.error(f"Error serializing results for {analysis_type} scoring: {e}. Results snippet: {str(results)[:200]}")
            return None # Serialization error means we can't proceed

        prompt = prompt_template.format(analysis_type=analysis_type, json_results=json_results)

        response = await self.llm.complete(prompt)
        if not response:
            logger.warning(f"LLM returned no response for {analysis_type} scoring. Defaulting to None.")
            return None
        try:
            score = float(response.strip())
            if 0 <= score <= 100:
                return score
            else:
                logger.warning(f"LLM score {score} for {analysis_type} is outside 0-100 range. Defaulting to None.")
                return None
        except ValueError:
            logger.warning(f"Could not convert LLM score response '{response.strip()}' to float for {analysis_type}. Defaulting to None.")
            return None
        except Exception as e: # Catch any other unexpected errors during scoring
            logger.error(f"Unexpected error in _score_analysis for {analysis_type}: {e}", exc_info=True)
            return None

    async def _score_fundamental_analysis(self, results: Dict[str, Any]) -> Optional[float]:
        return await self._score_analysis("fundamental", results)

    async def _score_technical_analysis(self, results: Dict[str, Any]) -> Optional[float]:
        return await self._score_analysis("technical", results)

    async def _score_sentiment_analysis(self, results: Dict[str, Any]) -> Optional[float]:
        return await self._score_analysis("sentiment", results)


    async def _generate_final_analysis(self,
                                     stock_data: StockData,
                                     results: Dict[str, Any],
                                     scores: Dict[str, Optional[float]], # Scores can be None
                                     analysis_type: AnalysisType) -> AnalysisResult:
        """Generate final analysis and recommendation, handling potential errors and None scores."""

        valid_scores = [s for s in scores.values() if s is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0 # Default if no valid scores

        # Prepare results for JSON dump, handling potential circular refs or unserializable data
        # This is for the summary prompt; individual helper functions will do their own error handling for LLM calls.
        try:
            results_for_summary = {k: v for k,v in results.items() if not k.endswith("_error") and k != "chain_execution_error"}
            # Sanitize further if necessary, e.g. convert complex objects to strings
            results_summary_json = json.dumps(results_for_summary, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error serializing results for final summary prompt of {stock_data.symbol}: {e}")
            results_summary_json = json.dumps({"error": "Could not serialize analysis results for summary."}, default=str)

        summary_prompt_parts = [
            f"Provide a comprehensive investment analysis summary for {stock_data.symbol} ({getattr(stock_data, 'company_name', 'N/A')})."
        ]
        if results.get("analysis_orchestration_error"):
             summary_prompt_parts.append(f"\nWARNING: The analysis process encountered a critical error: {results['analysis_orchestration_error']}")
        elif results.get("overall_analysis_status") == "Completed with errors":
            summary_prompt_parts.append("\nWARNING: Some parts of the analysis encountered errors. Results may be incomplete.")
        
        summary_prompt_parts.append(f"\nAnalysis Results (raw data, may be partial):\n{results_summary_json}")
        
        summary_prompt_parts.append(f"\nScores (0-100, N/A if not computed or error):")
        summary_prompt_parts.append(f"- Fundamental: {scores.get('fundamental', 'N/A')}")
        summary_prompt_parts.append(f"- Technical: {scores.get('technical', 'N/A')}")
        summary_prompt_parts.append(f"- Sentiment: {scores.get('sentiment', 'N/A')}")
        summary_prompt_parts.append(f"- Overall Calculated: {overall_score:.2f} (based on available scores)")
        
        summary_prompt_parts.append("""
        \nInstructions for LLM:
        1. Briefly (1-2 sentences) state the overall status of the analysis (e.g., successful, partial, errors encountered).
        2. Executive summary (2-3 sentences) of the investment profile.
        3. Key investment thesis (positive points, if data supports). If analysis failed or data is negative, state that.
        4. Major risks and concerns (negative points, if data supports).
        5. Briefly comment on the calculated overall score and what it implies, considering available data.
        
        Format as a clear, professional analysis. If data is largely missing or erroneous, explicitly state that the analysis is inconclusive.
        """)
        summary_prompt = "\n".join(summary_prompt_parts)

        summary = await self.llm.complete(summary_prompt)
        if not summary:
            summary = "Could not generate final summary due to an LLM error or insufficient data. Check logs."
            logger.warning(f"LLM call for final summary failed for {stock_data.symbol}. Prompt sent: {summary_prompt[:500]}...")

        # Default values for parts that rely on LLM calls that might fail or return unusable data
        bull_case_default = ["Could not generate bull case due to error or insufficient data."]
        bear_case_default = ["Could not generate bear case due to error or insufficient data."]
        
        # Only attempt bull/bear if some analysis was successful
        bull_bear_data = {'bull_case': bull_case_default, 'bear_case': bear_case_default}
        if not results.get("analysis_orchestration_error") and results.get("overall_analysis_status") not in ["No analysis performed (all data skipped)", "Pending"]:
            bull_bear_prompt = f"""
            Based on the preceding analysis summary and data for {stock_data.symbol}, provide:

            BULL CASE (3-5 concise bullet points, if supported by the data):

            BEAR CASE (3-5 concise bullet points, if supported by the data):

            Format strictly as JSON:
            {{
                "bull_case": ["point1", "point2", ...],
                "bear_case": ["point1", "point2", ...]
            }}
            If data is insufficient for a robust case for either, return an empty list for that case (e.g., "bull_case": []).
            """
            # Pass the generated summary as context, along with original results if needed (though summary should encapsulate)
            # For brevity, we'll primarily rely on the LLM's understanding from the summary_prompt context if it was extensive.
            # A better approach might be to re-include key parts of `results_summary_json` here too.

            bull_bear_response = await self.llm.complete(bull_bear_prompt) # Potentially add `system_prompt` too
            if bull_bear_response:
                parsed_bb = self.llm.parse_json_response(bull_bear_response)
                if parsed_bb and isinstance(parsed_bb.get('bull_case'), list) and isinstance(parsed_bb.get('bear_case'), list):
                    bull_bear_data = parsed_bb
                else:
                    logger.warning(f"Failed to parse bull/bear case JSON correctly for {stock_data.symbol}. Response: {bull_bear_response[:200]}")
            else:
                logger.warning(f"LLM call for bull/bear case failed for {stock_data.symbol}.")
        else:
             logger.warning(f"Skipping bull/bear case generation for {stock_data.symbol} due to prior analysis errors or no data.")


        recommendation = self._calculate_recommendation(overall_score)
        confidence = self._calculate_confidence(scores, results)

        current_price_default = getattr(stock_data, 'current_price', 0.0) if stock_data else 0.0
        price_target_val = await self._calculate_price_target(stock_data, results, default_target=current_price_default)

        risk_factors_val = await self._identify_risk_factors(stock_data, results)
        catalysts_val = await self._identify_catalysts(results)

        return AnalysisResult(
            stock=stock_data,
            analysis_type=analysis_type,
            llm_summary=summary,
            bull_case=bull_bear_data.get('bull_case', bull_case_default),
            bear_case=bull_bear_data.get('bear_case', bear_case_default),
            fundamental_score=scores.get('fundamental'), # Will be None if error/skipped
            technical_score=scores.get('technical'),
            sentiment_score=scores.get('sentiment'),
            overall_score=overall_score,
            recommendation=recommendation,
            confidence=confidence,
            price_target=price_target_val,
            risk_factors=risk_factors_val,
            catalysts=catalysts_val
        )

    def _calculate_recommendation(self, score: float) -> Recommendation:
        """Convert score to recommendation"""
        if score >= 80:
            return Recommendation.STRONG_BUY
        elif score >= 65:
            return Recommendation.BUY
        elif score >= 35:
            return Recommendation.HOLD
        elif score >= 20:
            return Recommendation.SELL
        else:
            return Recommendation.STRONG_SELL

    def _calculate_confidence(self, scores: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate confidence level"""
        """Calculate confidence level based on score consistency and error presence."""
        # Filter out None scores before calculation
        valid_scores = [s for s in scores.values() if s is not None]

        if not valid_scores: # No valid scores to base confidence on
            confidence = 0.1 # Very low
        else:
            avg_score = sum(valid_scores) / len(valid_scores)
            if len(valid_scores) > 1:
                variance = sum((x - avg_score) ** 2 for x in valid_scores) / len(valid_scores)
                std_dev = variance ** 0.5
                # Normalize std_dev effect: higher std_dev means lower confidence
                # Max std_dev could be around 50 (e.g. scores 0, 100).
                # (1 - std_dev / 50) maps 0 std_dev to 1 confidence, 50 std_dev to 0 confidence.
                confidence = max(0.0, 1.0 - (std_dev / 50.0))
            else: # Only one score available
                confidence = 0.5 # Medium confidence if only one score type is available but valid

        # Reduce confidence if there were errors in the results or critical failures
        if results and (any(key.endswith("_error") or "chain_execution_error" in key for key in results.keys()) or results.get("analysis_orchestration_error")):
            confidence *= 0.5 # Penalize for any errors

        # Reduce confidence if not all analysis types were performed (e.g. only fundamental, not comprehensive)
        num_expected_scores = len(scores) # Total number of score types defined
        num_valid_scores = len(valid_scores)
        if num_valid_scores < num_expected_scores:
            confidence *= (0.5 + 0.5 * (num_valid_scores / num_expected_scores)) # Scale down if partial

        return round(max(0.1, min(0.9, confidence)), 2) # Bound confidence between 0.1 and 0.9

    async def _calculate_price_target(self, stock_data: StockData, results: Dict[str, Any], default_target: Optional[float] = None) -> Optional[float]:
        """Calculate 12-month price target. Returns None if unable to calculate."""
        stock_symbol = getattr(stock_data, 'symbol', 'Unknown Stock')
        current_price = getattr(stock_data, 'current_price', None)

        if default_target is None: # If no explicit default, use current price if available
            default_target = current_price

        if not results or results.get("analysis_orchestration_error") or all(k.endswith("_error") or k.endswith("_skipped") for k in results if k not in ['symbol', 'company_name', 'stock_data', 'overall_analysis_status']): # Check if all actual analyses failed/skipped
            logger.warning(f"Skipping price target calculation for {stock_symbol} due to widespread errors or no analysis in results.")
            return default_target

        try:
            # Use only relevant, non-error parts of results for the prompt
            results_for_prompt = {k:v for k,v in results.items() if not k.endswith("_error") and k != "chain_execution_error" and not k.endswith("_skipped")}
            if not results_for_prompt:
                 logger.warning(f"No valid analysis results to use for price target for {stock_symbol}.")
                 return default_target
            results_json = json.dumps(results_for_prompt, indent=2, default=str)
        except TypeError as e:
            logger.error(f"Error serializing results for price target calculation of {stock_symbol}: {e}")
            return default_target

        prompt = f"""
        Based on the analysis of {stock_symbol} (current price: ${current_price if current_price is not None else 'N/A'}):
        {results_json}
        
        Provide a 12-month price target. Consider all available data:
        - Current valuation metrics, growth prospects, industry multiples, technical levels.
        
        Respond with ONLY a single numerical value (e.g., 125.50).
        If data is clearly insufficient or analysis was negative, you can respond with the current price or a conservatively adjusted price.
        """

        response = await self.llm.complete(prompt)
        if not response:
            logger.warning(f"LLM returned no response for price target of {stock_symbol}. Defaulting.")
            return default_target
        try:
            return float(response.strip())
        except ValueError:
            logger.warning(f"Could not convert LLM price target response '{response.strip()}' to float for {stock_symbol}. Defaulting.")
            return default_target
        except Exception as e:
            logger.error(f"Unexpected error in _calculate_price_target for {stock_symbol}: {e}", exc_info=True)
            return default_target


    async def _identify_list_items(self, item_type: str, stock_data: StockData, results: Dict[str, Any], default_items: List[str]) -> List[str]:
        """Generic helper to identify list items like risks or catalysts."""
        stock_symbol = getattr(stock_data, 'symbol', 'Unknown Stock')
        if not results or results.get("analysis_orchestration_error") or all(k.endswith("_error") or k.endswith("_skipped") for k in results if k not in ['symbol', 'company_name', 'stock_data', 'overall_analysis_status']):
            logger.warning(f"Skipping {item_type} identification for {stock_symbol} due to errors/no analysis in results.")
            return default_items[:1] + ["Analysis incomplete or errors encountered."]


        try:
            results_for_prompt = {k:v for k,v in results.items() if not k.endswith("_error") and k != "chain_execution_error" and not k.endswith("_skipped")}
            if not results_for_prompt:
                 logger.warning(f"No valid analysis results to use for {item_type} identification for {stock_symbol}.")
                 return default_items
            results_json = json.dumps(results_for_prompt, indent=2, default=str)
        except TypeError as e:
            logger.error(f"Error serializing results for {item_type} identification of {stock_symbol}: {e}")
            return default_items

        prompt = f"""
        Based on the analysis of {stock_symbol}:
        {results_json}
        
        Identify the top 3-5 {item_type}.
        Format as a JSON array of strings: ["item1", "item2", ...].
        If data is insufficient, return an empty array or generic {item_type}.
        """

        response = await self.llm.complete(prompt)
        if not response:
            logger.warning(f"LLM returned no response for {item_type} of {stock_symbol}.")
            return default_items

        parsed = self.llm.parse_json_response(response)

        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed if parsed else default_items # Return parsed if not empty, else default
        # Sometimes LLM might return {"item_type": ["item1", ...]}
        elif isinstance(parsed, dict) and isinstance(parsed.get(item_type), list) and all(isinstance(item, str) for item in parsed.get(item_type, [])):
             return parsed.get(item_type, default_items)

        logger.warning(f"Could not parse {item_type} for {stock_symbol} from LLM response: {response[:100]}... Returning defaults.")
        return default_items

    async def _identify_risk_factors(self, stock_data: StockData, results: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        default_risks = ["General market volatility", "Sector-specific risks", "Company execution risks"]
        return await self._identify_list_items("risk factors", stock_data, results, default_risks)

    async def _identify_catalysts(self, results: Dict[str, Any]) -> List[str]: # StockData may not be strictly needed if results are comprehensive
        """Identify potential catalysts"""
        # Assuming 'stock_data' (symbol, etc.) is already part of 'results' or not critical for this specific prompt
        # For consistency, one might add stock_data as an argument if prompts are adapted.
        # For now, let's make a dummy stock_data if it were needed by _identify_list_items's current signature
        # This is a bit of a workaround for the current signature of _identify_list_items.
        # Ideally, _identify_catalysts would also take stock_data.
        dummy_stock_data_for_catalysts = StockData(symbol=results.get('symbol', 'Unknown')) if 'symbol' in results else StockData(symbol='N/A')

        default_catalysts = ["Positive earnings surprises", "New product innovations", "Favorable market trends"]
        return await self._identify_list_items("potential positive catalysts", dummy_stock_data_for_catalysts, results, default_catalysts)


    # Serialization helpers
    def _serialize_stock_data(self, stock: Optional[StockData]) -> Dict[str, Any]:
        """Convert StockData to dict for LLM, handling potential None values and missing object."""
        if not stock:
            return {'error': 'StockData object is None', 'symbol': 'N/A'}
        try:
            # Helper to safely get attributes
            def get_attr(obj, attr_name, default='N/A'):
                val = getattr(obj, attr_name, default)
                return val if val is not None else default

            return {
                'symbol': get_attr(stock, 'symbol'),
                'price': get_attr(stock, 'current_price'),
                'market_cap': get_attr(stock, 'market_cap'),
                'pe_ratio': get_attr(stock, 'pe_ratio'),
                'dividend_yield': get_attr(stock, 'dividend_yield'),
                'eps': get_attr(stock, 'eps'),
                'revenue': get_attr(stock, 'revenue'),
                'profit_margin': get_attr(stock, 'profit_margin'),
                'debt_to_equity': get_attr(stock, 'debt_to_equity'),
                'beta': get_attr(stock, 'beta'),
                'year_high': get_attr(stock, 'year_high'),
                'year_low': get_attr(stock, 'year_low'),
                # Ensure all fields from StockData model are included if they are used by LLM
                'sector': get_attr(stock, 'sector'),
                'industry': get_attr(stock, 'industry'),
                'country': get_attr(stock, 'country'),
                'description': get_attr(stock, 'description', '')[:200] + "..." # Truncate long descriptions
            }
        except AttributeError as e: # Should be largely covered by get_attr, but as a fallback
            logger.error(f"Error serializing StockData for {getattr(stock, 'symbol', 'Unknown Symbol')}: {e}")
            return {'symbol': getattr(stock, 'symbol', 'Unknown Symbol'), 'error': f'Failed to serialize stock data: {e}'}
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error serializing StockData for {getattr(stock, 'symbol', 'Unknown Symbol')}: {e}", exc_info=True)
            return {'symbol': getattr(stock, 'symbol', 'Unknown Symbol'), 'error': f'Unexpected error during serialization: {e}'}


    def _serialize_financial_metrics(self, metrics: Optional[FinancialMetrics]) -> Dict[str, Any]:
        """Convert FinancialMetrics to dict for LLM, handling None and missing attributes."""
        if not metrics:
            return {'error': 'Financial metrics not available or object is None.'}
        try:
            # Assuming FinancialMetrics has attributes like 'ratios', 'growth_rates', etc.
            # If these are complex objects themselves, they might need their own serializers or careful handling.
            return {
                'ratios': getattr(metrics, 'ratios', {}) if getattr(metrics, 'ratios', {}) is not None else {},
                'growth_rates': getattr(metrics, 'growth_rates', {}) if getattr(metrics, 'growth_rates', {}) is not None else {},
                'peer_comparison': getattr(metrics, 'peer_comparison', {}) if getattr(metrics, 'peer_comparison', {}) is not None else {}
                # Add other relevant fields from FinancialMetrics model
            }
        except AttributeError as e:
            logger.error(f"Error serializing FinancialMetrics: {e}")
            return {'error': f'Failed to serialize financial metrics due to missing attribute: {e}'}
        except Exception as e:
            logger.error(f"Unexpected error serializing FinancialMetrics: {e}", exc_info=True)
            return {'error': f'Unexpected error during FinancialMetrics serialization: {e}'}

    def _serialize_technical_indicators(self, indicators: Optional[TechnicalIndicators]) -> Dict[str, Any]:
        """Convert TechnicalIndicators to dict for LLM, handling None and missing attributes."""
        if not indicators:
            return {'error': 'Technical indicators not available or object is None.'}
        try:
            # Helper to safely get attributes from indicators object
            def get_ind_attr(attr_name, default_val='N/A'):
                val = getattr(indicators, attr_name, default_val)
                # Handle nested dicts like macd or moving_averages if they could be None
                if isinstance(default_val, dict) and val is None:
                    return {}
                return val if val is not None else default_val

            return {
                'rsi': get_ind_attr('rsi'),
                'macd': get_ind_attr('macd', {}), # MACD might be a dict e.g. {'macd': val, 'signal': val}
                'moving_averages': get_ind_attr('moving_averages', {}), # e.g. {'sma50': val, 'ema20': val}
                'trend_direction': get_ind_attr('trend_direction'), # Renamed from 'trend' for clarity
                'momentum_score': get_ind_attr('momentum_score'), # Renamed from 'momentum'
                'support_levels': get_ind_attr('support_levels', []), # Example: if you add this
                'resistance_levels': get_ind_attr('resistance_levels', []) # Example: if you add this
            }
        except AttributeError as e:
            logger.error(f"Error serializing TechnicalIndicators: {e}")
            return {'error': f'Failed to serialize technical indicators due to missing attribute: {e}'}
        except Exception as e:
            logger.error(f"Unexpected error serializing TechnicalIndicators: {e}", exc_info=True)
            return {'error': f'Unexpected error during TechnicalIndicators serialization: {e}'}


    def _serialize_news_item(self, news: Optional[NewsItem]) -> Dict[str, Any]:
        """Convert NewsItem to dict for LLM, handling None and missing or invalid attributes."""
        if not news:
            return {'error': 'NewsItem object is None', 'title': 'N/A'}
        try:
            published_date = getattr(news, 'published', None)
            if isinstance(published_date, datetime):
                published_iso = published_date.isoformat()
            elif published_date is not None: # If it's some other format, try to convert or log warning
                logger.warning(f"NewsItem published date for '{getattr(news, 'title', 'Unknown Title')}' is not a datetime object: {published_date}. Storing as string.")
                published_iso = str(published_date)
            else:
                published_iso = 'N/A'

            return {
                'title': getattr(news, 'title', 'N/A'),
                'source': getattr(news, 'source', 'N/A'),
                'summary': getattr(news, 'summary', 'N/A')[:500] + "..." if getattr(news, 'summary', None) else 'N/A', # Truncate long summaries
                'sentiment_score': getattr(news, 'sentiment_score', 'N/A'), # Assuming this is already a simple type
                'published': published_iso,
                'url': getattr(news, 'url', 'N/A') # If you have a URL field
            }
        except AttributeError as e:
            logger.error(f"Error serializing NewsItem (title: {getattr(news, 'title', 'Unknown Title')}): {e}")
            return {'title': getattr(news, 'title', 'N/A'), 'error': f'Failed to serialize news item due to missing attribute: {e}'}
        except Exception as e:
            logger.error(f"Unexpected error serializing NewsItem (title: {getattr(news, 'title', 'N/A')}): {e}", exc_info=True)
            return {'title': getattr(news, 'title', 'N/A'), 'error': f'Unexpected error during NewsItem serialization: {e}'}

class NaturalLanguageScreener:
    """Convert natural language queries to stock filters"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language screening query"""

        system_prompt = """
        You are a financial query parser. Convert natural language stock screening queries into structured filters.
        
        Extract:
        - Quantitative criteria (price, market cap, ratios)
        - Qualitative criteria (sector, growth characteristics)
        - Sorting preferences
        - Any specific stock mentions
        
        Output as JSON with the following structure:
        {
            "filters": [
                {"field": "price", "operator": "<", "value": 50},
                {"field": "pe_ratio", "operator": "<", "value": 20}
            ],
            "qualitative": {
                "sector": "technology",
                "characteristics": ["growing earnings", "low debt"]
            },
            "sort_by": "market_cap",
            "sort_order": "desc",
            "limit": 20
        }
        """

        try:
            if not query:
                logger.warning("NaturalLanguageScreener.parse_query received an empty query.")
                return {"error": "Query cannot be empty.", "filters": []} # Return a structure that callers might expect

            response = await self.llm.complete(
                f"Parse this stock screening query: {query}",
                system_prompt=system_prompt,
                temperature=0.2 # Lower temperature for more deterministic parsing
            )
            if not response: # LLM call itself failed or returned empty
                logger.error(f"LLM returned no response for query parsing. Query: {query}")
                return {"error": "LLM failed to provide a response for query parsing.", "filters": []}

            parsed_response = self.llm.parse_json_response(response)
            if not parsed_response: # JSON parsing failed
                 logger.warning(f"Failed to parse LLM JSON response for query: '{query}'. Response: {response[:200]}...")
                 return {"error": "Failed to parse LLM's structured response for the query.", "filters": []}

            # Basic validation of parsed structure (optional, but good practice)
            if "filters" not in parsed_response and "qualitative" not in parsed_response:
                logger.warning(f"Parsed query response for '{query}' lacks expected structure. Got: {parsed_response}")
                # return {"error": "Parsed query lacks expected filter structure.", "filters": []}
                # Depending on strictness, you might still return parsed_response if partially valid
            return parsed_response

        except Exception as e:
            logger.error(f"Error parsing natural language query '{query}': {e}", exc_info=True)
            return {"error": f"Failed to parse query due to an unexpected error: {e}", "filters": []}


    async def explain_matches(self, query: str, matched_stocks: List[StockData]) -> Dict[str, str]:
        """Explain why each stock matched the query, with improved error handling and data serialization."""

        explanations = {}
        if not matched_stocks:
            logger.info("explain_matches received no stocks to explain.")
            return explanations

        if not query:
            logger.warning("explain_matches received an empty query string.")
            for stock in matched_stocks[:10]:
                 if stock and hasattr(stock, 'symbol'):
                    explanations[stock.symbol] = "Cannot explain match: Query is empty."
            return explanations


        for stock_idx, stock in enumerate(matched_stocks[:10]):  # Limit to top 10 to manage API calls & time
            stock_symbol = getattr(stock, 'symbol', f'UnknownStock_{stock_idx}')
            try:
                if not stock: # Should not happen if matched_stocks contains valid StockData objects
                    logger.warning(f"Skipping explanation for null stock object at index {stock_idx}.")
                    explanations[stock_symbol] = "Error: Stock data is missing."
                    continue

                # Use the robust serialization method
                stock_data_dict = self._serialize_stock_data(stock)
                if 'error' in stock_data_dict:
                    logger.warning(f"Failed to serialize stock data for {stock_symbol} for explanation: {stock_data_dict['error']}")
                    explanations[stock_symbol] = f"Error: Could not prepare stock data for {stock_symbol}."
                    continue

                # Create a concise summary from serialized data for the prompt
                # Avoid sending overly verbose data if not needed for this specific prompt
                prompt_stock_data = (
                    f"- Symbol: {stock_data_dict.get('symbol', 'N/A')}\n"
                    f"- Price: ${stock_data_dict.get('price', 'N/A')}\n"
                    f"- Market Cap: ${stock_data_dict.get('market_cap', 'N/A'):,}\n" # Format market cap if it's a number
                    f"- P/E Ratio: {stock_data_dict.get('pe_ratio', 'N/A')}\n"
                    f"- Sector: {stock_data_dict.get('sector', 'N/A')}"
                )


                prompt = f"""
                Explain concisely (1-2 sentences) why {stock_data_dict.get('company_name', stock_symbol)} matches the screening query:
                "{query}"

                Relevant Stock data:
                {prompt_stock_data}

                Focus on the most direct reasons for the match based on the query.
                """

                explanation_response = await self.llm.complete(prompt, temperature=0.4) # Slightly lower temp for factual explanation
                if explanation_response:
                    explanations[stock_symbol] = explanation_response.strip()
                else:
                    logger.warning(f"LLM returned no explanation for {stock_symbol} and query '{query}'.")
                    explanations[stock_symbol] = "Could not generate explanation (LLM returned empty)."
            
            except AttributeError as e:
                logger.error(f"Attribute error processing stock {stock_symbol} for explanation: {e}", exc_info=True)
                explanations[stock_symbol] = "Error generating explanation due to missing stock data attribute."
            except Exception as e:
                logger.error(f"Error explaining match for stock {stock_symbol}: {e}", exc_info=True)
                explanations[stock_symbol] = f"Error generating explanation for {stock_symbol}."
                # Optionally, re-raise if it's critical, or continue to next stock

        return explanations
