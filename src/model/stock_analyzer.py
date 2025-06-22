import json
from typing import Dict, Optional, Any

# --- Enhanced AI Analyzer ---
import openai
import requests
import re

from src.model.llm_utils import LLM
from src.model.stock_data import StockData
import logging

# --- logging -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StockAnalyzer:
    def __init__(self):
        self.llm = LLM()

    def analyze_stock(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze stock with or without AI, return structured JSON"""
        summary = stock_data.get_summary()

        if not self.llm.has_ai:
            return self._basic_analysis_json(summary)

        try:
            if self.llm.url:
                return self._lm_studio_analysis(summary)
            else:
                return self._openai_analysis(summary)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._basic_analysis_json(summary)

    def _openai_analysis(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI for analysis"""
        prompt = self._create_analysis_prompt(summary)

        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a stock analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse OpenAI response as JSON")
            return self._basic_analysis_json(summary)

    def _lm_studio_analysis(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use LM Studio for analysis"""
        prompt = self._create_analysis_prompt(summary)

        try:
            response = requests.post(
                f"{self.llm.url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a stock analyst. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Clean and parse JSON
                try:
                    # Remove potential markdown code blocks
                    cleaned_content = re.sub(r'```json|```', '', content).strip()
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LM Studio response as JSON")
                    return self._basic_analysis_json(summary)
            else:
                logger.error(f"LM Studio error: {response.status_code} - {response.text}")
                return self._basic_analysis_json(summary)

        except Exception as e:
            logger.error(f"LM Studio connection error: {e}")
            return self._basic_analysis_json(summary)

    def _create_analysis_prompt(self, summary: Dict[str, Any]) -> str:
        """Create analysis prompt for LLM"""
        return f"""
        Analyze this stock and provide a structured JSON response:
        
        Stock: {summary['symbol']} - {summary['name']}
        Price: ${summary['price']} ({summary['change']['percent']:+.2f}%)
        Market Cap: ${summary['market_cap']:,}
        P/E Ratio: {summary['pe_ratio']}
        52-Week Range: ${summary['52w_low']} - ${summary['52w_high']}
        Sector: {summary['sector']}
        
        Respond with ONLY a JSON object in this exact format:
        {{
            "recommendation": "BUY/HOLD/SELL",
            "confidence": 0-100,
            "price_target": number,
            "risk_level": "LOW/MEDIUM/HIGH",
            "summary": "Brief analysis summary",
            "strengths": ["strength1", "strength2", "strength3"],
            "risks": ["risk1", "risk2", "risk3"],
            "technical_indicators": {{
                "trend": "BULLISH/BEARISH/NEUTRAL",
                "support_level": number,
                "resistance_level": number
            }},
            "valuation": {{
                "assessment": "UNDERVALUED/FAIR/OVERVALUED",
                "reasoning": "Brief explanation"
            }}
        }}
        """

    def _basic_analysis_json(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Basic rule-based analysis in JSON format"""
        # Price position calculation
        price_position = 0.5
        if summary['52w_high'] > 0 and summary['52w_low'] > 0:
            price_position = (summary['price'] - summary['52w_low']) / (summary['52w_high'] - summary['52w_low'])

        # P/E based valuation
        pe = summary['pe_ratio']
        if pe > 0:
            if pe < 15:
                valuation = "UNDERVALUED"
                valuation_reason = "Low P/E ratio suggests undervaluation"
            elif pe > 30:
                valuation = "OVERVALUED"
                valuation_reason = "High P/E ratio suggests overvaluation"
            else:
                valuation = "FAIR"
                valuation_reason = "P/E ratio within normal range"
        else:
            valuation = "UNKNOWN"
            valuation_reason = "No P/E ratio available"

        # Trend determination
        trend = "NEUTRAL"
        if summary['change']['percent'] > 2:
            trend = "BULLISH"
        elif summary['change']['percent'] < -2:
            trend = "BEARISH"

        # Recommendation logic
        if valuation == "UNDERVALUED" and trend != "BEARISH":
            recommendation = "BUY"
            confidence = 75
        elif valuation == "OVERVALUED" and trend != "BULLISH":
            recommendation = "SELL"
            confidence = 70
        else:
            recommendation = "HOLD"
            confidence = 60

        # Risk assessment
        if price_position > 0.8:
            risk_level = "HIGH"
        elif price_position < 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "price_target": round(summary['price'] * 1.1, 2),  # Simple 10% upside
            "risk_level": risk_level,
            "summary": f"{summary['name']} is currently {valuation.lower()} with a {trend.lower()} trend.",
            "strengths": [
                f"Trading at ${summary['price']}" + (" near 52-week low" if price_position < 0.2 else ""),
                f"Market cap of ${summary['market_cap']:,.0f}",
                f"In the {summary['sector']} sector" if summary['sector'] != 'Unknown' else "Established company"
            ],
            "risks": [
                "Near 52-week high" if price_position > 0.8 else "Market volatility",
                f"P/E ratio of {pe:.1f}" if pe > 25 else "Sector competition",
                "Limited dividend yield" if summary['dividend_yield'] < 1 else "Economic uncertainty"
            ],
            "technical_indicators": {
                "trend": trend,
                "support_level": round(summary['52w_low'], 2),
                "resistance_level": round(summary['52w_high'], 2)
            },
            "valuation": {
                "assessment": valuation,
                "reasoning": valuation_reason
            }
        }
