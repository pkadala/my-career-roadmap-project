#!/usr/bin/env python3
"""
Career Roadmap AI - Complete Project Generator (Part 1)
This script generates the complete project with all code.
"""
import os
import json
import textwrap
from pathlib import Path


def create_file(filepath: str, content: str):
    """Create a file with the given content."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")


def create_base_files():
    """Create base configuration files."""
    
    # .gitignore
    create_file('.gitignore', """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# IDE
.idea/
*.swp
*.swo

# Environment
.env
.env.local
.env.*.local

# Database
*.db
*.sqlite3
data/vector_store/

# Logs
logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# OS
.DS_Store
Thumbs.db

# Docker
docker/data/
""")

    # requirements.txt
    create_file('requirements.txt', """# Core dependencies
langchain==0.3.25
langchain-openai==0.3.0
langchain-anthropic==0.3.0
langchain-community==0.3.25
langchain-core==0.3.25

# API framework
fastapi==0.115.5
uvicorn==0.32.0
pydantic==2.10.0
pydantic-settings==2.6.0

# Database and caching
sqlalchemy==2.0.36
alembic==1.14.0
redis==5.2.0
asyncpg==0.30.0

# Vector store
chromadb==0.5.0
faiss-cpu==1.9.0

# Utilities
python-dotenv==1.0.1
httpx==0.28.0
tenacity==9.0.0
python-multipart==0.0.12
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Monitoring and logging
structlog==24.4.0
prometheus-client==0.21.0

# Data processing
pandas==2.2.3
numpy==2.0.2
scikit-learn==1.5.2

# Development tools
pytest==8.3.4
pytest-asyncio==0.25.0
black==24.10.0
flake8==7.1.1
mypy==1.13.0
""")

    # .env.example
    create_file('.env.example', """# Application Settings
APP_NAME="Career Roadmap AI"
APP_VERSION="1.0.0"
DEBUG=false
API_PREFIX="/api/v1"

# Security
SECRET_KEY="your-secret-key-here-change-this-in-production"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Database
DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/career_roadmap"
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL="redis://localhost:6379"
CACHE_TTL=3600

# LLM Configuration
LLM_PROVIDER="openai"
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"

# Model Settings
OPENAI_MODEL="gpt-4-turbo-preview"
ANTHROPIC_MODEL="claude-3-opus-20240229"
TEMPERATURE=0.7
MAX_TOKENS=4000

# Vector Store
VECTOR_STORE_PATH="./data/vector_store"
EMBEDDING_MODEL="text-embedding-3-small"

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Logging
LOG_LEVEL="INFO"
LOG_FORMAT="json"

# Feature Flags
ENABLE_CACHING=true
ENABLE_ANALYTICS=true
ENABLE_PROGRESS_TRACKING=true
ENABLE_RATE_LIMITING=true

# External APIs (Optional)
COURSERA_API_KEY=""
UDEMY_API_KEY=""
LINKEDIN_API_KEY=""
""")

    # pyproject.toml
    create_file('pyproject.toml', """[tool.black]
line-length = 120
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 120

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
""")


def create_app_layer():
    """Create the app layer files."""
    
    # app/__init__.py
    create_file('app/__init__.py', '')
    
    # app/config.py
    create_file('app/config.py', '''"""
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
    llm_provider: str = "openai"  # "openai" or "anthropic"
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
''')

    # app/main.py
    create_file('app/main.py', '''"""
Main FastAPI application for Career Roadmap AI.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from app.config import settings
from app.middleware import setup_middleware
from api.v1.router import api_router
from utils.logger import get_logger
from services.database_service import init_db
from services.cache_service import init_cache

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle events."""
    # Startup
    logger.info("Starting Career Roadmap AI application...")
    await init_db()
    await init_cache()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Career Roadmap AI application...")
    # Cleanup resources
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup middleware
setup_middleware(app)

# Include API router
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Career Roadmap AI",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app_version
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id if hasattr(request.state, "request_id") else None
        }
    )
''')

    # app/middleware.py
    create_file('app/middleware.py', '''"""
Middleware configuration for the FastAPI application.
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import uuid
from typing import Callable
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware (security)
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.yourdomain.com", "yourdomain.com"]
        )
    
    # Custom middleware
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(logging_middleware)
    app.middleware("http")(timing_middleware)
    
    # Rate limiting middleware
    if settings.enable_rate_limiting:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log all requests and responses."""
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None)
    )
    
    response = await call_next(request)
    
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        request_id=getattr(request.state, "request_id", None)
    )
    
    return response


async def timing_middleware(request: Request, call_next: Callable) -> Response:
    """Add request timing information."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        logger.warning(
            "Slow request detected",
            method=request.method,
            path=request.url.path,
            process_time=process_time,
            request_id=getattr(request.state, "request_id", None)
        )
    
    return response
''')

    # app/dependencies.py
    create_file('app/dependencies.py', '''"""
FastAPI dependencies for authentication and common functionality.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from app.config import settings

security = HTTPBearer()


def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token_data: dict = Depends(verify_token)) -> Optional[str]:
    """Get current user from token."""
    # In production, this would fetch user from database
    # For now, return user_id from token
    return token_data.get("sub", "anonymous")


# For development/testing without authentication
async def get_current_user_optional() -> Optional[str]:
    """Get current user optionally (for development)."""
    return "test-user-123"
''')


def create_api_layer():
    """Create the API layer files."""
    
    # Create all __init__.py files
    init_files = [
        'api/__init__.py',
        'api/v1/__init__.py',
        'api/v1/endpoints/__init__.py',
        'api/models/__init__.py'
    ]
    
    for init_file in init_files:
        create_file(init_file, '')
    
    # api/v1/router.py
    create_file('api/v1/router.py', '''"""
API v1 router configuration.
"""
from fastapi import APIRouter
from api.v1.endpoints import roadmap, career_analysis, skills, progress

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(roadmap.router, prefix="/roadmap", tags=["roadmap"])
api_router.include_router(career_analysis.router, prefix="/career", tags=["career"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(progress.router, prefix="/progress", tags=["progress"])
''')

    # api/models/requests.py
    create_file('api/models/requests.py', '''"""
API request models for the Career Roadmap AI application.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    EXPERT = "expert"


class CareerTransitionRequest(BaseModel):
    """Request model for career transition analysis."""
    current_role: str = Field(..., description="Current job title/role")
    target_role: str = Field(..., description="Target job title/role")
    current_skills: List[str] = Field(default=[], description="List of current skills")
    experience_years: int = Field(..., ge=0, le=50, description="Years of experience")
    experience_level: ExperienceLevel = Field(..., description="Current experience level")
    available_hours_per_week: int = Field(..., ge=1, le=100, description="Hours available for learning per week")
    preferred_learning_style: Optional[str] = Field(None, description="Preferred learning style")
    budget_constraint: Optional[float] = Field(None, ge=0, description="Monthly budget for learning")
    target_timeline_months: Optional[int] = Field(None, ge=1, le=60, description="Target timeline in months")
    industry_preference: Optional[str] = Field(None, description="Preferred industry")
    location: Optional[str] = Field(None, description="Location for job market analysis")
    
    @validator('current_skills')
    def validate_skills(cls, v):
        return [skill.strip() for skill in v if skill.strip()]


class SkillsAnalysisRequest(BaseModel):
    """Request model for skills gap analysis."""
    current_role: str
    target_role: str
    current_skills: List[str]
    include_soft_skills: bool = True
    include_technical_skills: bool = True


class ProgressUpdateRequest(BaseModel):
    """Request model for progress updates."""
    roadmap_id: str
    completed_items: List[str]
    current_skill_levels: Dict[str, int]  # skill_name: proficiency_level (1-10)
    feedback: Optional[str] = None
''')

    # api/models/responses.py
    create_file('api/models/responses.py', '''"""
API response models for the Career Roadmap AI application.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SkillAssessment(BaseModel):
    """Skill assessment details."""
    skill_name: str
    current_level: int = Field(..., ge=0, le=10)
    required_level: int = Field(..., ge=0, le=10)
    gap_size: int
    transferable: bool
    priority: str  # "high", "medium", "low"
    related_skills: List[str] = []


class LearningResource(BaseModel):
    """Learning resource details."""
    resource_id: str
    title: str
    type: str  # "course", "book", "project", "certification", "article"
    provider: str
    url: Optional[str] = None
    duration_hours: int
    difficulty_level: str
    cost: float = 0.0
    rating: Optional[float] = None
    skills_covered: List[str]
    prerequisites: List[str] = []


class Milestone(BaseModel):
    """Roadmap milestone."""
    milestone_id: str
    title: str
    description: str
    target_date: datetime
    skills_to_achieve: List[str]
    resources: List[LearningResource]
    projects: List[str]
    estimated_hours: int
    checkpoint_criteria: List[str]


class RoadmapMetrics(BaseModel):
    """Roadmap metrics and estimates."""
    total_hours_required: int
    estimated_completion_months: float
    difficulty_rating: float = Field(..., ge=1, le=10)
    confidence_score: float = Field(..., ge=0, le=1)
    total_cost_estimate: float
    job_market_demand: str  # "high", "medium", "low"
    success_probability: float = Field(..., ge=0, le=1)


class CareerTransitionRoadmap(BaseModel):
    """Complete career transition roadmap response."""
    roadmap_id: str
    created_at: datetime
    current_role: str
    target_role: str
    
    # Skills analysis
    transferable_skills: List[SkillAssessment]
    skills_to_acquire: List[SkillAssessment]
    skill_gap_summary: str
    
    # Learning path
    milestones: List[Milestone]
    recommended_resources: List[LearningResource]
    recommended_projects: List[Dict[str, Any]]
    certifications: List[Dict[str, Any]]
    
    # Metrics and estimates
    metrics: RoadmapMetrics
    
    # Personalized insights
    personalized_advice: str
    potential_challenges: List[str]
    success_factors: List[str]
    alternative_paths: List[Dict[str, Any]]


class ProgressReport(BaseModel):
    """Progress tracking report."""
    roadmap_id: str
    overall_progress_percentage: float
    completed_milestones: int
    total_milestones: int
    skills_acquired: List[str]
    current_phase: str
    estimated_completion_date: datetime
    recommendations: List[str]
    achievement_badges: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
''')


def main():
    """Main function to generate part 1 of the project."""
    print("🚀 Generating Career Roadmap AI Project - Part 1\n")
    
    # Create base files
    create_base_files()
    
    # Create app layer
    create_app_layer()
    
    # Create API layer
    create_api_layer()
    
    print("\n✅ Part 1 completed! Run generate_project_part2.py next.")


if __name__ == "__main__":
    main()
