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

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
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

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
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
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logger.error(f"Gemini API error: {response.status}")
                        return ""

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
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
                        return data[0]['generated_text']
                    else:
                        logger.error(f"HuggingFace API error: {response.status}")
                        return ""

        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
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
                        return data['choices'][0]['message']['content']
                    else:
                        logger.error(f"LM Studio API error: {response.status}")
                        return ""

        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            return ""

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
            return {}

# Import aiohttp for async HTTP requests
try:
    import aiohttp
except ImportError:
    logger.warning("aiohttp not installed, some LLM providers may not work")
    aiohttp = None

class ChainPromptExecutor:
    """Executes multi-step chain prompts"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.context = {}

    async def execute_chain(self, template: ChainPromptTemplate, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of prompts"""
        self.context = initial_context.copy()
        results = {}

        for step in template.steps:
            # Format prompt with current context
            prompt = self._format_prompt(step['prompt'], self.context)

            # Execute LLM call
            response = await self.llm.complete(prompt)

            # Store result
            output_key = step['output']
            results[output_key] = response
            self.context[output_key] = response

            # Parse structured output if needed
            if template.output_format == "json":
                parsed = self.llm.parse_json_response(response)
                if parsed:
                    self.context[f"{output_key}_parsed"] = parsed
                    results[f"{output_key}_parsed"] = parsed

        return results

    def _format_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Format prompt template with context values"""
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            template = template.replace(f"{{{key}}}", str(value))
        return template

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

        # Run appropriate analysis chains
        if analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.COMPREHENSIVE]:
            if financial_metrics:
                context['financial_data'] = self._serialize_financial_metrics(financial_metrics)
                fundamental_results = await self._run_fundamental_analysis(context)
                results.update(fundamental_results)
                scores['fundamental'] = await self._score_fundamental_analysis(fundamental_results)

        if analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPREHENSIVE]:
            if technical_indicators:
                context['technical_indicators'] = self._serialize_technical_indicators(technical_indicators)
                technical_results = await self._run_technical_analysis(context)
                results.update(technical_results)
                scores['technical'] = await self._score_technical_analysis(technical_results)

        if analysis_type in [AnalysisType.SENTIMENT, AnalysisType.COMPREHENSIVE]:
            if news_items:
                context['news_data'] = [self._serialize_news_item(item) for item in news_items]
                sentiment_results = await self._run_sentiment_analysis(context)
                results.update(sentiment_results)
                scores['sentiment'] = await self._score_sentiment_analysis(sentiment_results)

        # Generate final analysis
        analysis_result = await self._generate_final_analysis(stock_data, results, scores)

        return analysis_result

    async def _run_fundamental_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run fundamental analysis chain"""
        # Add peer data (mock for now)
        context['peer_list'] = ['AAPL', 'MSFT', 'GOOGL']  # Would get actual peers
        context['peer_data'] = {}  # Would fetch actual peer data
        context['industry_trends'] = "Technology sector showing strong growth"  # Would fetch real trends

        results = await self.chain_executor.execute_chain(FUNDAMENTAL_ANALYSIS_CHAIN, context)
        return results

    async def _run_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run technical analysis chain"""
        # Prepare price and volume data
        context['price_data'] = context['technical_indicators'].get('moving_averages', {})
        context['volume_data'] = context['technical_indicators'].get('volume_profile', {})

        results = await self.chain_executor.execute_chain(TECHNICAL_ANALYSIS_CHAIN, context)
        return results

    async def _run_sentiment_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run news sentiment analysis chain"""
        results = await self.chain_executor.execute_chain(NEWS_SENTIMENT_CHAIN, context)
        return results

    async def _score_fundamental_analysis(self, results: Dict[str, Any]) -> float:
        """Score fundamental analysis results"""
        prompt = f"""
        Based on the following fundamental analysis, provide a score from 0-100:
        {json.dumps(results, indent=2)}
        
        Consider:
        - Financial health and stability
        - Growth prospects
        - Competitive position
        - Valuation metrics
        
        Respond with just a number between 0-100.
        """

        response = await self.llm.complete(prompt)
        try:
            return float(response.strip())
        except:
            return 50.0

    async def _score_technical_analysis(self, results: Dict[str, Any]) -> float:
        """Score technical analysis results"""
        prompt = f"""
        Based on the following technical analysis, provide a score from 0-100:
        {json.dumps(results, indent=2)}
        
        Consider:
        - Trend strength and direction
        - Momentum indicators
        - Support/resistance levels
        - Volume patterns
        
        Respond with just a number between 0-100.
        """

        response = await self.llm.complete(prompt)
        try:
            return float(response.strip())
        except:
            return 50.0

    async def _score_sentiment_analysis(self, results: Dict[str, Any]) -> float:
        """Score sentiment analysis results"""
        prompt = f"""
        Based on the following sentiment analysis, provide a score from 0-100:
        {json.dumps(results, indent=2)}
        
        Consider:
        - Overall news sentiment
        - Impact of recent events
        - Market perception
        - Future catalysts
        
        Respond with just a number between 0-100.
        """

        response = await self.llm.complete(prompt)
        try:
            return float(response.strip())
        except:
            return 50.0

    async def _generate_final_analysis(self,
                                     stock_data: StockData,
                                     results: Dict[str, Any],
                                     scores: Dict[str, float]) -> AnalysisResult:
        """Generate final analysis and recommendation"""

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 50.0

        # Generate comprehensive summary
        summary_prompt = f"""
        Provide a comprehensive investment analysis summary for {stock_data.symbol} ({stock_data.company_name}).
        
        Analysis Results:
        {json.dumps(results, indent=2)}
        
        Scores:
        - Fundamental: {scores.get('fundamental', 'N/A')}
        - Technical: {scores.get('technical', 'N/A')}
        - Sentiment: {scores.get('sentiment', 'N/A')}
        
        Provide:
        1. Executive summary (2-3 sentences)
        2. Key investment thesis
        3. Major risks and concerns
        4. Price target rationale
        
        Format as a clear, professional analysis.
        """

        summary = await self.llm.complete(summary_prompt)

        # Extract bull and bear cases
        bull_bear_prompt = f"""
        Based on the analysis of {stock_data.symbol}, provide:
        
        BULL CASE (3-5 points):
        - Strong positive factors
        
        BEAR CASE (3-5 points):
        - Key risks and negatives
        
        Format as JSON:
        {{
            "bull_case": ["point1", "point2", ...],
            "bear_case": ["point1", "point2", ...]
        }}
        """

        bull_bear_response = await self.llm.complete(bull_bear_prompt)
        bull_bear_data = self.llm.parse_json_response(bull_bear_response)

        # Determine recommendation
        recommendation = self._calculate_recommendation(overall_score)

        # Calculate confidence
        confidence = self._calculate_confidence(scores, results)

        # Create analysis result
        return AnalysisResult(
            stock=stock_data,
            analysis_type=AnalysisType.COMPREHENSIVE,
            llm_summary=summary,
            bull_case=bull_bear_data.get('bull_case', []),
            bear_case=bull_bear_data.get('bear_case', []),
            fundamental_score=scores.get('fundamental', 0),
            technical_score=scores.get('technical', 0),
            sentiment_score=scores.get('sentiment', 0),
            overall_score=overall_score,
            recommendation=recommendation,
            confidence=confidence,
            price_target=await self._calculate_price_target(stock_data, results),
            risk_factors=await self._identify_risk_factors(stock_data, results),
            catalysts=await self._identify_catalysts(results)
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
        # Base confidence on score consistency
        if not scores:
            return 0.5

        score_values = list(scores.values())
        avg_score = sum(score_values) / len(score_values)

        # Calculate standard deviation
        variance = sum((x - avg_score) ** 2 for x in score_values) / len(score_values)
        std_dev = variance ** 0.5

        # Lower std dev = higher confidence
        confidence = max(0.3, min(0.9, 1 - (std_dev / 50)))

        return confidence

    async def _calculate_price_target(self, stock_data: StockData, results: Dict[str, Any]) -> float:
        """Calculate 12-month price target"""
        prompt = f"""
        Based on the analysis of {stock_data.symbol} (current price: ${stock_data.current_price}):
        {json.dumps(results, indent=2)}
        
        Provide a 12-month price target. Consider:
        - Current valuation metrics
        - Growth prospects
        - Industry multiples
        - Technical levels
        
        Respond with just a number (e.g., 125.50).
        """

        response = await self.llm.complete(prompt)
        try:
            return float(response.strip())
        except:
            return stock_data.current_price * 1.1  # Default to 10% upside

    async def _identify_risk_factors(self, stock_data: StockData, results: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        prompt = f"""
        Identify the top 3-5 risk factors for {stock_data.symbol}:
        {json.dumps(results, indent=2)}
        
        Format as JSON array of strings.
        """

        response = await self.llm.complete(prompt)
        parsed = self.llm.parse_json_response(response)

        if isinstance(parsed, list):
            return parsed
        return ["Market volatility", "Competition", "Regulatory changes"]

    async def _identify_catalysts(self, results: Dict[str, Any]) -> List[str]:
        """Identify potential catalysts"""
        prompt = f"""
        Based on the analysis, identify 3-5 potential positive catalysts:
        {json.dumps(results, indent=2)}
        
        Format as JSON array of strings.
        """

        response = await self.llm.complete(prompt)
        parsed = self.llm.parse_json_response(response)

        if isinstance(parsed, list):
            return parsed
        return ["Earnings growth", "New product launches", "Market expansion"]

    # Serialization helpers
    def _serialize_stock_data(self, stock: StockData) -> Dict[str, Any]:
        """Convert StockData to dict for LLM"""
        return {
            'symbol': stock.symbol,
            'price': stock.current_price,
            'market_cap': stock.market_cap,
            'pe_ratio': stock.pe_ratio,
            'dividend_yield': stock.dividend_yield,
            'eps': stock.eps,
            'revenue': stock.revenue,
            'profit_margin': stock.profit_margin,
            'debt_to_equity': stock.debt_to_equity,
            'beta': stock.beta,
            'year_high': stock.year_high,
            'year_low': stock.year_low
        }

    def _serialize_financial_metrics(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """Convert FinancialMetrics to dict for LLM"""
        return {
            'ratios': metrics.ratios,
            'growth_rates': metrics.growth_rates,
            'peer_comparison': metrics.peer_comparison
        }

    def _serialize_technical_indicators(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Convert TechnicalIndicators to dict for LLM"""
        return {
            'rsi': indicators.rsi,
            'macd': indicators.macd,
            'moving_averages': indicators.moving_averages,
            'trend': indicators.trend_direction,
            'momentum': indicators.momentum_score
        }

    def _serialize_news_item(self, news: NewsItem) -> Dict[str, Any]:
        """Convert NewsItem to dict for LLM"""
        return {
            'title': news.title,
            'source': news.source,
            'summary': news.summary,
            'sentiment': news.sentiment_score,
            'published': news.published.isoformat()
        }

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

        response = await self.llm.complete(
            f"Parse this stock screening query: {query}",
            system_prompt=system_prompt,
            temperature=0.3
        )

        return self.llm.parse_json_response(response)

    async def explain_matches(self, query: str, matched_stocks: List[StockData]) -> Dict[str, str]:
        """Explain why each stock matched the query"""

        explanations = {}

        for stock in matched_stocks[:10]:  # Limit to top 10
            prompt = f"""
            Explain in one sentence why {stock.symbol} ({stock.company_name}) matches this query:
            "{query}"
            
            Stock data:
            - Price: ${stock.current_price}
            - Market Cap: ${stock.market_cap:,.0f}
            - P/E Ratio: {stock.pe_ratio}
            - Sector: {stock.sector}
            """

            explanation = await self.llm.complete(prompt, temperature=0.5)
            explanations[stock.symbol] = explanation.strip()

        return explanations
