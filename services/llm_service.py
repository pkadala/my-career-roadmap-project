"""
Service for managing LLM interactions.
"""
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MockLLM(BaseChatModel):
    """Mock LLM for development when API keys are not available."""
    
    def __init__(self):
        super().__init__()
        logger.warning("Using MockLLM - API keys not configured")
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Return a mock response in the correct format
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        
        mock_text = "This is a mock response. Please configure your API keys to use the real LLM."
        generation = ChatGeneration(message=AIMessage(content=mock_text))
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "mock"


class LLMService:
    """Service for managing LLM interactions."""
    
    def __init__(self):
        self._llm_cache: Dict[str, BaseChatModel] = {}
    
    def get_llm(
        self, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> BaseChatModel:
        """Get configured LLM instance."""
        provider = provider or settings.llm_provider
        cache_key = f"{provider}:{model}:{temperature}:{max_tokens}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        if provider == "openai":
            if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
                logger.warning("OpenAI API key not configured, using mock LLM")
                llm = MockLLM()
            else:
                llm = ChatOpenAI(
                    model=model or settings.openai_model,
                    temperature=temperature or settings.temperature,
                    max_tokens=max_tokens or settings.max_tokens,
                    api_key=settings.openai_api_key,
                    timeout=30
                )
        elif provider == "anthropic":
            if not settings.anthropic_api_key or settings.anthropic_api_key == "your-anthropic-api-key-here":
                logger.warning("Anthropic API key not configured, using mock LLM")
                llm = MockLLM()
            else:
                llm = ChatAnthropic(
                    model=model or settings.anthropic_model,
                    temperature=temperature or settings.temperature,
                    max_tokens=max_tokens or settings.max_tokens,
                    anthropic_api_key=settings.anthropic_api_key,
                    timeout=30
                )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        self._llm_cache[cache_key] = llm
        logger.info(f"Created LLM instance: {provider} - {model}")
        return llm


# Singleton instance
llm_service = LLMService()

def get_llm(**kwargs) -> BaseChatModel:
    """Get LLM instance."""
    return llm_service.get_llm(**kwargs)
