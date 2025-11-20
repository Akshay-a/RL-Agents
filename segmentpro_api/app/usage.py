"""
Usage tracking and rate limiting
"""

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any

from . import models
from .config import PRICING_TIERS


def get_current_billing_period() -> str:
    """Get current billing period in YYYY-MM format"""
    return datetime.utcnow().strftime("%Y-%m")


def record_usage(
    db: Session,
    user_id: int,
    endpoint: str,
    method: str = "POST",
    processing_time_ms: int = 0,
    input_size_bytes: int = 0,
    output_size_bytes: int = 0
) -> models.UsageRecord:
    """Record an API usage event"""
    record = models.UsageRecord(
        user_id=user_id,
        endpoint=endpoint,
        method=method,
        api_calls=1,
        processing_time_ms=processing_time_ms,
        input_size_bytes=input_size_bytes,
        output_size_bytes=output_size_bytes,
        billing_period=get_current_billing_period()
    )
    db.add(record)
    db.commit()
    return record


def get_usage_stats(db: Session, user_id: int, user_tier: str) -> Dict[str, Any]:
    """Get user's usage statistics for current billing period"""
    billing_period = get_current_billing_period()

    # Get total API calls
    total_calls = db.query(func.sum(models.UsageRecord.api_calls)).filter(
        models.UsageRecord.user_id == user_id,
        models.UsageRecord.billing_period == billing_period
    ).scalar() or 0

    # Get calls by endpoint type
    image_calls = db.query(func.sum(models.UsageRecord.api_calls)).filter(
        models.UsageRecord.user_id == user_id,
        models.UsageRecord.billing_period == billing_period,
        models.UsageRecord.endpoint.like("%/segment/image%")
    ).scalar() or 0

    video_calls = db.query(func.sum(models.UsageRecord.api_calls)).filter(
        models.UsageRecord.user_id == user_id,
        models.UsageRecord.billing_period == billing_period,
        models.UsageRecord.endpoint.like("%/segment/video%")
    ).scalar() or 0

    batch_calls = db.query(func.sum(models.UsageRecord.api_calls)).filter(
        models.UsageRecord.user_id == user_id,
        models.UsageRecord.billing_period == billing_period,
        models.UsageRecord.endpoint.like("%/segment/batch%")
    ).scalar() or 0

    # Get tier limits
    tier_config = PRICING_TIERS.get(user_tier, PRICING_TIERS["free"])
    api_limit = tier_config["api_calls_monthly"] or float("inf")

    return {
        "billing_period": billing_period,
        "api_calls_used": int(total_calls),
        "api_calls_limit": api_limit if api_limit != float("inf") else -1,
        "api_calls_remaining": max(0, api_limit - total_calls) if api_limit != float("inf") else -1,
        "image_segmentations": int(image_calls),
        "video_segmentations": int(video_calls),
        "batch_jobs": int(batch_calls),
        "current_tier": user_tier,
        "estimated_cost": tier_config["price_monthly"] or 0
    }


def check_usage_limit(db: Session, user_id: int, user_tier: str) -> tuple[bool, str]:
    """
    Check if user has exceeded their usage limit.

    Returns:
        Tuple of (is_allowed, message)
    """
    stats = get_usage_stats(db, user_id, user_tier)

    # Enterprise users have unlimited access
    if user_tier == "enterprise":
        return True, "OK"

    # Check against limit
    if stats["api_calls_limit"] > 0 and stats["api_calls_used"] >= stats["api_calls_limit"]:
        return False, f"Monthly API limit reached ({stats['api_calls_limit']} calls). Please upgrade your plan."

    return True, "OK"
