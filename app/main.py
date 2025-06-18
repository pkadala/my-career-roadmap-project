"""
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
