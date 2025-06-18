"""
Logging configuration for the application.
"""
import logging
import structlog
from typing import Optional
from app.config import settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

'''
def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    logger = structlog.get_logger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    return logger
'''

def get_logger(name):
    std_logger = logging.getLogger(name)
    std_logger.setLevel(logging.DEBUG)  # ✅ Set level on std logger
    return structlog.wrap_logger(std_logger)