"""
Authentication and authorization utilities
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db
from . import models

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

settings = get_settings()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    return encoded_jwt


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"sp_{secrets.token_urlsafe(32)}"


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """Get current user from JWT token"""
    if credentials is None:
        return None

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    return user


async def get_current_user_from_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """Get current user from API key"""
    if api_key is None:
        return None

    api_key_record = db.query(models.APIKey).filter(
        models.APIKey.key == api_key,
        models.APIKey.is_active == True
    ).first()

    if api_key_record is None:
        return None

    # Update last used timestamp
    api_key_record.last_used_at = datetime.utcnow()
    db.commit()

    return api_key_record.user


async def get_current_user(
    token_user: Optional[models.User] = Depends(get_current_user_from_token),
    api_key_user: Optional[models.User] = Depends(get_current_user_from_api_key),
    db: Session = Depends(get_db)
) -> models.User:
    """Get current user from either JWT token or API key"""
    user = token_user or api_key_user

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user


async def get_optional_user(
    token_user: Optional[models.User] = Depends(get_current_user_from_token),
    api_key_user: Optional[models.User] = Depends(get_current_user_from_api_key),
) -> Optional[models.User]:
    """Get current user if authenticated, None otherwise"""
    return token_user or api_key_user


def check_tier_permission(user: models.User, required_tier: str) -> bool:
    """Check if user's tier allows access to a feature"""
    tier_hierarchy = {
        "free": 0,
        "starter": 1,
        "pro": 2,
        "business": 3,
        "enterprise": 4
    }

    user_level = tier_hierarchy.get(user.tier, 0)
    required_level = tier_hierarchy.get(required_tier, 0)

    return user_level >= required_level
