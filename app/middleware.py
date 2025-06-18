"""
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
        # allow_origins=settings.cors_origins,
        allow_origins=["*"],  # For development, use "*" to allow all origins
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
            # allowed_hosts=["*.yourdomain.com", "yourdomain.com"] This is for Prod # environments
            allowed_hosts=["*"]  # For development, use "*" to allow all hosts
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
