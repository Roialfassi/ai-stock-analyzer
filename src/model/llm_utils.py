import json
import requests
import re
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enum for available LLM providers"""
    OPENAI = "openai"
    LM_STUDIO = "lm_studio"
    ANTHROPIC = "anthropic"
    NONE = "none"


class LLMResponse:
    """Standardized response from LLM providers"""

    def __init__(self, content: str, success: bool, error: Optional[str] = None,
                 raw_response: Optional[Dict] = None):
        self.content = content
        self.success = success
        self.error = error
        self.raw_response = raw_response
        self._parsed_json = None

    def get_json(self) -> Optional[Dict]:
        """Parse content as JSON if possible"""
        if self._parsed_json is not None:
            return self._parsed_json

        if not self.success or not self.content:
            return None

        try:
            # Clean common JSON formatting issues
            cleaned = self.content.strip()
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*|\s*```', '', cleaned)
            # Remove any leading/trailing whitespace
            cleaned = cleaned.strip()

            self._parsed_json = json.loads(cleaned)
            return self._parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None,
              temperature: float = 0.7, max_tokens: int = 500) -> LLMResponse:
        """Send query to LLM and return response"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass


class LLM:
    has_ai: bool
    provider: LLMProvider
    key: str
    url: str

    def __init__(self, provider: LLMProvider = LLMProvider.NONE, has_ai: bool = False, url: str = "", key: str = ""):
        self.has_ai = has_ai
        self.provider = provider
        self.url = url
        self.key = key
