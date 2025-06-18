"""
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
