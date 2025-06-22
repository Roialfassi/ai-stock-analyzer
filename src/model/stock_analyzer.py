import json
from typing import Dict, Optional, Any
import openai
import requests
import re
import logging

from src.model.llm_utils import LLM
from src.model.stock_data import StockData

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
        system_prompt = self._get_system_prompt()

        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3  # Lower temperature for more consistent formatting
        )

        return self._parse_llm_response(response.choices[0].message.content, summary)

    def _lm_studio_analysis(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use LM Studio for analysis"""
        prompt = self._create_analysis_prompt(summary)
        system_prompt = self._get_system_prompt()

        try:
            response = requests.post(
                f"{self.llm.url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                },
                timeout=1000
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return self._parse_llm_response(content, summary)
            else:
                logger.error(f"LM Studio error: {response.status_code} - {response.text}")
                return self._basic_analysis_json(summary)

        except Exception as e:
            logger.error(f"LM Studio connection error: {e}")
            return self._basic_analysis_json(summary)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for consistent JSON output"""
        return """You are a professional stock analyst AI. Your responses must be ONLY valid JSON with no additional text, explanations, or markdown formatting.

CRITICAL RULES:
1. Output ONLY the JSON object - no text before or after
2. No markdown code blocks (no ```json or ```)
3. Use double quotes for all strings
4. Numbers should not have quotes
5. Arrays must use square brackets []
6. Ensure all required fields are present
7. Keep all text concise and professional"""

    def _create_analysis_prompt(self, summary: Dict[str, Any]) -> str:
        """Create enhanced analysis prompt for LLM"""
        return f"""Analyze this stock and provide your assessment as a JSON object:

STOCK DATA:
• Symbol: {summary['symbol']}
• Company: {summary['name']}
• Current Price: ${summary['price']:.2f}
• Day Change: {summary['change']['percent']:+.2f}%
• Market Cap: ${summary['market_cap']:,}
• P/E Ratio: {summary['pe_ratio']:.2f}
• 52-Week Low: ${summary['52w_low']:.2f}
• 52-Week High: ${summary['52w_high']:.2f}
• Volume: {summary['volume']:,}
• Dividend Yield: {summary['dividend_yield']:.2f}%
• Sector: {summary['sector']}
• Industry: {summary['industry']}

REQUIRED JSON FORMAT (copy this structure exactly):
{{
    "recommendation": "<Must be exactly one of: BUY, HOLD, SELL>",
    "confidence": <Integer from 0 to 100>,
    "price_target": <Realistic 12-month price target as decimal number>,
    "risk_level": "<Must be exactly one of: LOW, MEDIUM, HIGH>",
    "summary": "<1-2 sentence analysis summary. Be specific about this stock>",
    "strengths": [
        "<Specific strength 1>",
        "<Specific strength 2>",
        "<Specific strength 3>"
    ],
    "risks": [
        "<Specific risk 1>",
        "<Specific risk 2>",
        "<Specific risk 3>"
    ],
    "technical_indicators": {{
        "trend": "<Must be exactly one of: BULLISH, BEARISH, NEUTRAL>",
        "support_level": <Key support price as decimal>,
        "resistance_level": <Key resistance price as decimal>
    }},
    "valuation": {{
        "assessment": "<Must be exactly one of: UNDERVALUED, FAIR, OVERVALUED>",
        "reasoning": "<1 sentence explaining the valuation assessment>"
    }}
}}

ANALYSIS GUIDELINES:
- Recommendation should consider both fundamentals and current momentum
- Confidence reflects certainty in your recommendation (higher = more certain)
- Price target should be realistic based on current price and fundamentals
- Risk level should consider volatility, sector risks, and company specifics
- Strengths and risks must be specific to THIS company, not generic
- Support level should be below current price, resistance above
- Base valuation on P/E ratio, growth prospects, and sector comparisons

OUTPUT ONLY THE JSON OBJECT WITH NO ADDITIONAL TEXT."""

    def _parse_llm_response(self, content: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Flexible parser for LLM responses"""
        try:
            # Clean the response
            cleaned = content.strip()

            # Remove markdown code blocks if present
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

            # Remove any text before the first { and after the last }
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            # Try to parse the JSON
            result = json.loads(cleaned)

            # Validate and fix the structure
            return self._validate_and_fix_analysis(result, summary)

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw content: {content}")

            # Try to extract values using regex as fallback
            try:
                return self._regex_parse_response(content, summary)
            except Exception as parse_error:
                logger.error(f"Regex parsing also failed: {parse_error}")
                return self._basic_analysis_json(summary)

    def _validate_and_fix_analysis(self, data: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix the analysis data structure"""
        # Define valid values
        valid_recommendations = ["BUY", "HOLD", "SELL"]
        valid_risk_levels = ["LOW", "MEDIUM", "HIGH"]
        valid_trends = ["BULLISH", "BEARISH", "NEUTRAL"]
        valid_assessments = ["UNDERVALUED", "FAIR", "OVERVALUED"]

        # Fix recommendation
        if "recommendation" not in data or data["recommendation"] not in valid_recommendations:
            data["recommendation"] = "HOLD"

        # Fix confidence
        if "confidence" not in data or not isinstance(data["confidence"], (int, float)):
            data["confidence"] = 50
        else:
            data["confidence"] = max(0, min(100, int(data["confidence"])))

        # Fix price target
        if "price_target" not in data or not isinstance(data["price_target"], (int, float)):
            data["price_target"] = round(summary['price'] * 1.1, 2)
        else:
            data["price_target"] = round(float(data["price_target"]), 2)

        # Fix risk level
        if "risk_level" not in data or data["risk_level"] not in valid_risk_levels:
            data["risk_level"] = "MEDIUM"

        # Fix summary
        if "summary" not in data or not isinstance(data["summary"], str):
            data["summary"] = f"{summary['name']} analysis based on current market conditions."

        # Fix arrays
        for field in ["strengths", "risks"]:
            if field not in data or not isinstance(data[field], list) or len(data[field]) < 3:
                if field == "strengths":
                    data[field] = [
                        f"Operating in {summary['sector']} sector",
                        f"Current P/E ratio of {summary['pe_ratio']:.1f}",
                        "Established market presence"
                    ]
                else:
                    data[field] = [
                        "Market volatility",
                        "Sector competition",
                        "Economic uncertainty"
                    ]

        # Fix technical indicators
        if "technical_indicators" not in data or not isinstance(data["technical_indicators"], dict):
            data["technical_indicators"] = {}

        tech = data["technical_indicators"]
        if "trend" not in tech or tech["trend"] not in valid_trends:
            tech["trend"] = "BULLISH" if summary['change']['percent'] > 0 else "BEARISH"

        if "support_level" not in tech or not isinstance(tech["support_level"], (int, float)):
            tech["support_level"] = round(summary['52w_low'], 2)
        else:
            tech["support_level"] = round(float(tech["support_level"]), 2)

        if "resistance_level" not in tech or not isinstance(tech["resistance_level"], (int, float)):
            tech["resistance_level"] = round(summary['52w_high'], 2)
        else:
            tech["resistance_level"] = round(float(tech["resistance_level"]), 2)

        data["technical_indicators"] = tech

        # Fix valuation
        if "valuation" not in data or not isinstance(data["valuation"], dict):
            data["valuation"] = {}

        val = data["valuation"]
        if "assessment" not in val or val["assessment"] not in valid_assessments:
            val["assessment"] = "FAIR"

        if "reasoning" not in val or not isinstance(val["reasoning"], str):
            val["reasoning"] = "Based on current P/E ratio and market conditions"

        data["valuation"] = val

        return data

    def _regex_parse_response(self, content: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback regex parser for malformed responses"""
        result = self._basic_analysis_json(summary)

        # Try to extract recommendation
        rec_match = re.search(r'recommendation["\s:]+(["\']?)(BUY|HOLD|SELL)\1', content, re.IGNORECASE)
        if rec_match:
            result["recommendation"] = rec_match.group(2).upper()

        # Try to extract confidence
        conf_match = re.search(r'confidence["\s:]+(\d+)', content)
        if conf_match:
            result["confidence"] = min(100, max(0, int(conf_match.group(1))))

        # Try to extract price target
        price_match = re.search(r'price_target["\s:]+(\d+(?:\.\d+)?)', content)
        if price_match:
            result["price_target"] = round(float(price_match.group(1)), 2)

        # Try to extract risk level
        risk_match = re.search(r'risk_level["\s:]+(["\']?)(LOW|MEDIUM|HIGH)\1', content, re.IGNORECASE)
        if risk_match:
            result["risk_level"] = risk_match.group(2).upper()

        # Try to extract trend
        trend_match = re.search(r'trend["\s:]+(["\']?)(BULLISH|BEARISH|NEUTRAL)\1', content, re.IGNORECASE)
        if trend_match:
            result["technical_indicators"]["trend"] = trend_match.group(2).upper()

        # Try to extract valuation
        val_match = re.search(r'assessment["\s:]+(["\']?)(UNDERVALUED|FAIR|OVERVALUED)\1', content, re.IGNORECASE)
        if val_match:
            result["valuation"]["assessment"] = val_match.group(2).upper()

        return result

    def _basic_analysis_json(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced rule-based analysis in JSON format"""
        # Price position calculation
        price_position = 0.5
        if summary['52w_high'] > 0 and summary['52w_low'] > 0:
            price_range = summary['52w_high'] - summary['52w_low']
            if price_range > 0:
                price_position = (summary['price'] - summary['52w_low']) / price_range

        # P/E based valuation
        pe = summary['pe_ratio']
        if pe > 0:
            if pe < 15:
                valuation = "UNDERVALUED"
                valuation_reason = "Low P/E ratio compared to market average"
            elif pe > 30:
                valuation = "OVERVALUED"
                valuation_reason = "High P/E ratio indicates premium valuation"
            else:
                valuation = "FAIR"
                valuation_reason = "P/E ratio within typical market range"
        else:
            valuation = "FAIR"
            valuation_reason = "Unable to assess based on P/E ratio"

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
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"

        # Price target calculation
        if recommendation == "BUY":
            price_target = round(summary['price'] * 1.15, 2)
        elif recommendation == "SELL":
            price_target = round(summary['price'] * 0.9, 2)
        else:
            price_target = round(summary['price'] * 1.05, 2)

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "price_target": price_target,
            "risk_level": risk_level,
            "summary": f"{summary['name']} shows {trend.lower()} momentum and appears {valuation.lower()} at current levels.",
            "strengths": [
                f"Trading at ${summary['price']:.2f}" + (" near 52-week low" if price_position < 0.3 else ""),
                f"Market cap of ${summary['market_cap']:,.0f}" + (" (Large cap)" if summary['market_cap'] > 10000000000 else ""),
                f"Active in {summary['sector']} sector" if summary['sector'] != 'Unknown' else "Established market presence"
            ],
            "risks": [
                "Trading near 52-week high" if price_position > 0.7 else "Current market volatility",
                f"P/E ratio of {pe:.1f} above sector average" if pe > 25 else "Competitive sector dynamics",
                f"Low dividend yield of {summary['dividend_yield']:.2f}%" if summary['dividend_yield'] < 2 else "Interest rate sensitivity"
            ],
            "technical_indicators": {
                "trend": trend,
                "support_level": round(summary['52w_low'] * 1.05, 2),  # 5% above 52w low
                "resistance_level": round(summary['52w_high'] * 0.95, 2)  # 5% below 52w high
            },
            "valuation": {
                "assessment": valuation,
                "reasoning": valuation_reason
            }
        }
