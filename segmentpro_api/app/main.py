"""
SegmentPro API - Main FastAPI Application
SAM 3 Powered Object Segmentation SaaS
"""

import io
import uuid
from datetime import datetime, timedelta
from typing import Optional

import structlog
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from sqlalchemy.orm import Session

from . import models, schemas
from .auth import (
    get_current_user,
    get_password_hash,
    verify_password,
    create_access_token,
    generate_api_key,
    check_tier_permission
)
from .config import get_settings, PRICING_TIERS
from .database import get_db, create_tables
from .sam3_service import get_sam3_service
from .usage import record_usage, get_usage_stats, check_usage_limit

logger = structlog.get_logger()
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="SAM 3 Powered Object Segmentation API - Segment anything with text prompts",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on startup"""
    create_tables()
    logger.info("Database tables created")

    # Pre-load SAM 3 model
    sam3 = get_sam3_service()
    logger.info("SAM 3 service initialized", model_loaded=sam3.is_loaded)


# Health check
@app.get("/health", response_model=schemas.HealthCheck, tags=["System"])
async def health_check():
    """Check API health status"""
    sam3 = get_sam3_service()
    import torch

    return {
        "status": "healthy",
        "version": settings.app_version,
        "model_loaded": sam3.is_loaded,
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "running"
    }


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/auth/register", response_model=schemas.UserResponse, tags=["Authentication"])
async def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user account"""
    # Check if email exists
    existing = db.query(models.User).filter(models.User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user = models.User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        company=user_data.company,
        tier="free"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create default API key
    api_key = models.APIKey(
        user_id=user.id,
        key=generate_api_key(),
        name="Default"
    )
    db.add(api_key)
    db.commit()

    logger.info("User registered", user_id=user.id, email=user.email)
    return user


@app.post("/auth/login", response_model=schemas.Token, tags=["Authentication"])
async def login(email: str, password: str, db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=schemas.UserResponse, tags=["Authentication"])
async def get_me(current_user: models.User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user


# =============================================================================
# API Key Management
# =============================================================================

@app.post("/api-keys", response_model=schemas.APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    key_data: schemas.APIKeyCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    api_key = models.APIKey(
        user_id=current_user.id,
        key=generate_api_key(),
        name=key_data.name
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return api_key


@app.get("/api-keys", response_model=list[schemas.APIKeyResponse], tags=["API Keys"])
async def list_api_keys(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys for current user"""
    keys = db.query(models.APIKey).filter(
        models.APIKey.user_id == current_user.id
    ).all()
    return keys


@app.delete("/api-keys/{key_id}", tags=["API Keys"])
async def delete_api_key(
    key_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an API key"""
    api_key = db.query(models.APIKey).filter(
        models.APIKey.id == key_id,
        models.APIKey.user_id == current_user.id
    ).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    db.delete(api_key)
    db.commit()

    return {"status": "deleted"}


# =============================================================================
# Segmentation Endpoints
# =============================================================================

@app.post("/segment/image", response_model=schemas.SegmentationResult, tags=["Segmentation"])
async def segment_image(
    request: schemas.SegmentImageRequest,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Segment objects in an image using text or geometric prompts.

    Examples:
    - Text prompt: "red car", "person wearing hat", "product on white background"
    - Box prompt: Provide bounding box coordinates [x1, y1, x2, y2]
    - Point prompt: Provide points and labels for foreground/background
    """
    # Check usage limits
    allowed, message = check_usage_limit(db, current_user.id, current_user.tier)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    # Get SAM 3 service
    sam3 = get_sam3_service()

    # TODO: Fetch image from URL
    # For now, create a placeholder image
    from PIL import Image
    image = Image.new("RGB", (512, 512), color="white")

    # Run segmentation
    result = sam3.segment_image(
        image=image,
        prompt=request.prompt,
        prompt_type=request.prompt_type.value,
        box=request.box,
        points=request.points,
        point_labels=request.point_labels,
        multimask_output=request.multimask_output,
        mask_threshold=request.mask_threshold
    )

    # Record usage
    background_tasks.add_task(
        record_usage,
        db, current_user.id, "/segment/image", "POST",
        result.get("processing_time_ms", 0), 0, 0
    )

    return result


@app.post("/segment/image/upload", response_model=schemas.SegmentationResult, tags=["Segmentation"])
async def segment_image_upload(
    file: UploadFile = File(...),
    prompt: str = "main object",
    prompt_type: str = "text",
    background_tasks: BackgroundTasks = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Segment objects in an uploaded image.

    Upload an image file directly and provide a text prompt.
    """
    # Check usage limits
    allowed, message = check_usage_limit(db, current_user.id, current_user.tier)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    # Check file size based on tier
    tier_config = PRICING_TIERS.get(current_user.tier, PRICING_TIERS["free"])
    max_size = tier_config["max_image_size_mb"] * 1024 * 1024

    # Read image
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {tier_config['max_image_size_mb']}MB for your tier"
        )

    # Load image
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

    # Get SAM 3 service
    sam3 = get_sam3_service()

    # Run segmentation
    result = sam3.segment_image(
        image=image,
        prompt=prompt,
        prompt_type=prompt_type
    )

    # Record usage
    if background_tasks:
        background_tasks.add_task(
            record_usage,
            db, current_user.id, "/segment/image/upload", "POST",
            result.get("processing_time_ms", 0), len(contents), 0
        )

    return result


@app.post("/segment/video", response_model=schemas.SegmentationResult, tags=["Segmentation"])
async def segment_video(
    request: schemas.SegmentVideoRequest,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Segment and track objects in a video.

    Provide a video URL and text prompt to track objects across frames.
    Requires Starter tier or higher.
    """
    # Check tier permission
    if not check_tier_permission(current_user, "starter"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Video segmentation requires Starter tier or higher"
        )

    # Check usage limits
    allowed, message = check_usage_limit(db, current_user.id, current_user.tier)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    # Get SAM 3 service
    sam3 = get_sam3_service()

    # Run video segmentation
    result = sam3.segment_video(
        video_path=request.video_url,
        prompt=request.prompt,
        start_frame=request.start_frame,
        end_frame=request.end_frame,
        track_objects=request.track_objects
    )

    # Record usage
    background_tasks.add_task(
        record_usage,
        db, current_user.id, "/segment/video", "POST",
        result.get("processing_time_ms", 0), 0, 0
    )

    return result


@app.post("/segment/batch", response_model=schemas.JobStatusResponse, tags=["Segmentation"])
async def segment_batch(
    request: schemas.BatchSegmentRequest,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Batch segment multiple images with the same prompt.

    Requires Pro tier or higher. Jobs are processed asynchronously.
    """
    # Check tier permission
    if not check_tier_permission(current_user, "pro"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Batch processing requires Pro tier or higher"
        )

    # Check usage limits
    allowed, message = check_usage_limit(db, current_user.id, current_user.tier)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    # Create batch job
    job_id = str(uuid.uuid4())
    job = models.SegmentationJob(
        job_id=job_id,
        user_id=current_user.id,
        job_type="batch",
        status="pending",
        prompt=request.prompt
    )
    db.add(job)
    db.commit()

    # TODO: Queue batch processing job

    return {
        "job_id": job_id,
        "status": schemas.JobStatus.PENDING,
        "progress": 0.0
    }


@app.get("/jobs/{job_id}", response_model=schemas.JobStatusResponse, tags=["Jobs"])
async def get_job_status(
    job_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a segmentation job"""
    job = db.query(models.SegmentationJob).filter(
        models.SegmentationJob.job_id == job_id,
        models.SegmentationJob.user_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": 1.0 if job.status == "completed" else 0.5,
        "result": None  # TODO: Return result when completed
    }


# =============================================================================
# Usage & Billing Endpoints
# =============================================================================

@app.get("/usage", response_model=schemas.UsageStats, tags=["Billing"])
async def get_usage(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current usage statistics"""
    stats = get_usage_stats(db, current_user.id, current_user.tier)
    return stats


@app.get("/pricing", tags=["Billing"])
async def get_pricing():
    """Get available pricing tiers"""
    return PRICING_TIERS


@app.post("/subscription", tags=["Billing"])
async def update_subscription(
    update: schemas.SubscriptionUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update subscription tier.

    In production, this would integrate with Stripe for payment processing.
    """
    if update.tier not in PRICING_TIERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {update.tier}"
        )

    # TODO: Stripe integration for actual payment
    # For demo, just update the tier
    current_user.tier = update.tier
    db.commit()

    return {
        "status": "updated",
        "tier": update.tier,
        "message": f"Subscription updated to {PRICING_TIERS[update.tier]['name']}"
    }


# =============================================================================
# Webhook Endpoints
# =============================================================================

@app.post("/webhooks", response_model=schemas.WebhookResponse, tags=["Webhooks"])
async def create_webhook(
    webhook: schemas.WebhookCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a webhook endpoint for job notifications"""
    if not check_tier_permission(current_user, "pro"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhooks require Pro tier or higher"
        )

    import json
    endpoint = models.WebhookEndpoint(
        user_id=current_user.id,
        url=webhook.url,
        events=json.dumps(webhook.events)
    )
    db.add(endpoint)
    db.commit()
    db.refresh(endpoint)

    return {
        "id": endpoint.id,
        "url": endpoint.url,
        "events": webhook.events,
        "is_active": endpoint.is_active,
        "created_at": endpoint.created_at
    }


# =============================================================================
# Model Info Endpoints
# =============================================================================

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the SAM 3 model"""
    sam3 = get_sam3_service()
    return sam3.get_model_info()


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
