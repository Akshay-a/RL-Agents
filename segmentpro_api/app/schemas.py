"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# Enums
class PromptType(str, Enum):
    TEXT = "text"
    BOX = "box"
    POINT = "point"
    MASK = "mask"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    PNG = "png"
    WEBP = "webp"
    JSON = "json"


# Auth Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    company: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    company: Optional[str]
    tier: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class APIKeyCreate(BaseModel):
    name: str = "Default"


class APIKeyResponse(BaseModel):
    id: int
    key: str
    name: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]

    class Config:
        from_attributes = True


# Segmentation Schemas
class SegmentImageRequest(BaseModel):
    """Request for image segmentation"""
    image_url: Optional[str] = None  # URL to image
    prompt: str = Field(..., description="Text description of what to segment")
    prompt_type: PromptType = PromptType.TEXT

    # Optional parameters
    box: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    points: Optional[List[List[float]]] = Field(None, description="Points [[x, y], ...]")
    point_labels: Optional[List[int]] = Field(None, description="Point labels (1=foreground, 0=background)")

    # Output options
    output_format: OutputFormat = OutputFormat.PNG
    return_mask: bool = True
    return_box: bool = True
    return_score: bool = True

    # Processing options
    multimask_output: bool = False
    mask_threshold: float = 0.5


class SegmentVideoRequest(BaseModel):
    """Request for video segmentation"""
    video_url: str
    prompt: str
    prompt_type: PromptType = PromptType.TEXT

    # Video specific
    start_frame: int = 0
    end_frame: Optional[int] = None
    track_objects: bool = True

    # Output options
    output_format: str = "mp4"  # mp4, frames, json


class BatchSegmentRequest(BaseModel):
    """Request for batch image segmentation"""
    images: List[str]  # List of image URLs
    prompt: str
    prompt_type: PromptType = PromptType.TEXT
    output_format: OutputFormat = OutputFormat.PNG


class SegmentationResult(BaseModel):
    """Result from segmentation"""
    job_id: str
    status: JobStatus

    # Results (populated when completed)
    masks: Optional[List[str]] = None  # Base64 encoded or URLs
    boxes: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    objects_detected: Optional[int] = None

    # Metadata
    processing_time_ms: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    # Error
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Job status check response"""
    job_id: str
    status: JobStatus
    progress: Optional[float] = None
    result: Optional[SegmentationResult] = None


# Usage & Billing Schemas
class UsageStats(BaseModel):
    """User's usage statistics"""
    billing_period: str
    api_calls_used: int
    api_calls_limit: int
    api_calls_remaining: int

    # Detailed breakdown
    image_segmentations: int
    video_segmentations: int
    batch_jobs: int

    # Costs
    current_tier: str
    estimated_cost: float


class SubscriptionUpdate(BaseModel):
    """Update subscription tier"""
    tier: str
    payment_method_id: Optional[str] = None


# Webhook Schemas
class WebhookCreate(BaseModel):
    url: str
    events: List[str] = ["job.completed", "job.failed"]


class WebhookResponse(BaseModel):
    id: int
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Health Check
class HealthCheck(BaseModel):
    status: str = "healthy"
    version: str
    model_loaded: bool
    gpu_available: bool
