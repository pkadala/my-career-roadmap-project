"""
Configuration management for the Career Roadmap AI application.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application settings
    app_name: str = "Career Roadmap AI"
    app_version: str = "1.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database
    database_url: str
    database_pool_size: int = 20
    database_max_overflow: int = 40
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # LLM Configuration
    llm_provider: str = "anthropic"  # "openai" or "anthropic"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model settings
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Vector store
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "text-embedding-3-small"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    enable_rate_limiting: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Feature flags
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_progress_tracking: bool = True
    
    # External APIs (optional)
    coursera_api_key: Optional[str] = None
    udemy_api_key: Optional[str] = None
    linkedin_api_key: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Initialize settings
settings = get_settings()
