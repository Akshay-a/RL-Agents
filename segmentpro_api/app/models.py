"""
Database models for SegmentPro API
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class TierEnum(str, enum.Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class User(Base):
    """User account model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    company = Column(String(255))

    # Subscription
    tier = Column(String(50), default="free")
    stripe_customer_id = Column(String(255))
    stripe_subscription_id = Column(String(255))

    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user")
    usage_records = relationship("UsageRecord", back_populates="user")


class APIKey(Base):
    """API key for authentication"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(255), default="Default")

    # Permissions
    is_active = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class UsageRecord(Base):
    """Track API usage for billing"""
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Request details
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)

    # Usage metrics
    api_calls = Column(Integer, default=1)
    processing_time_ms = Column(Integer)
    input_size_bytes = Column(Integer)
    output_size_bytes = Column(Integer)

    # Billing period
    billing_period = Column(String(7))  # Format: YYYY-MM

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_records")


class SegmentationJob(Base):
    """Track segmentation jobs"""
    __tablename__ = "segmentation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Job details
    job_type = Column(String(50), nullable=False)  # image, video, batch
    status = Column(String(50), default="pending")  # pending, processing, completed, failed

    # Input
    input_url = Column(Text)
    prompt = Column(Text)
    prompt_type = Column(String(50))  # text, box, point, mask

    # Output
    output_url = Column(Text)
    result_data = Column(Text)  # JSON string with masks, boxes, scores

    # Metrics
    processing_time_ms = Column(Integer)
    objects_detected = Column(Integer)

    # Error handling
    error_message = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class WebhookEndpoint(Base):
    """User-defined webhook endpoints"""
    __tablename__ = "webhook_endpoints"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    url = Column(Text, nullable=False)
    secret = Column(String(64))

    # Events to subscribe to
    events = Column(Text)  # JSON array of event types

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
