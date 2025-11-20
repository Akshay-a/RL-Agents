"""
Configuration management for SegmentPro API
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # App Config
    app_name: str = "SegmentPro API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./segmentpro.db"

    # Redis (for rate limiting and caching)
    redis_url: str = "redis://localhost:6379"

    # JWT Auth
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Stripe
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None

    # SAM 3 Model Config
    sam3_model_path: str = "./models/sam3"
    sam3_checkpoint: str = "sam3_large"
    device: str = "cuda"  # or "cpu"

    # Storage (S3 compatible)
    storage_bucket: str = "segmentpro-uploads"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"

    # Rate Limiting
    rate_limit_enabled: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Pricing Tiers Configuration
PRICING_TIERS = {
    "free": {
        "name": "Free",
        "price_monthly": 0,
        "api_calls_monthly": 100,
        "max_image_size_mb": 5,
        "max_video_length_seconds": 10,
        "features": [
            "Image segmentation",
            "Text prompts",
            "PNG/JPEG export",
            "Community support"
        ],
        "stripe_price_id": None
    },
    "starter": {
        "name": "Starter",
        "price_monthly": 19,
        "api_calls_monthly": 1000,
        "max_image_size_mb": 10,
        "max_video_length_seconds": 30,
        "features": [
            "Everything in Free",
            "Video segmentation",
            "Priority processing",
            "Email support"
        ],
        "stripe_price_id": "price_starter_monthly"
    },
    "pro": {
        "name": "Pro",
        "price_monthly": 49,
        "api_calls_monthly": 5000,
        "max_image_size_mb": 25,
        "max_video_length_seconds": 120,
        "features": [
            "Everything in Starter",
            "Batch processing",
            "Custom model fine-tuning",
            "Webhook notifications",
            "Priority support"
        ],
        "stripe_price_id": "price_pro_monthly"
    },
    "business": {
        "name": "Business",
        "price_monthly": 149,
        "api_calls_monthly": 25000,
        "max_image_size_mb": 50,
        "max_video_length_seconds": 300,
        "features": [
            "Everything in Pro",
            "Dedicated GPU instances",
            "SLA guarantee",
            "Custom integrations",
            "Dedicated support"
        ],
        "stripe_price_id": "price_business_monthly"
    },
    "enterprise": {
        "name": "Enterprise",
        "price_monthly": None,  # Custom pricing
        "api_calls_monthly": None,  # Unlimited
        "max_image_size_mb": 100,
        "max_video_length_seconds": None,
        "features": [
            "Everything in Business",
            "Unlimited API calls",
            "On-premise deployment",
            "Custom SLA",
            "24/7 support",
            "Dedicated account manager"
        ],
        "stripe_price_id": "price_enterprise_monthly"
    }
}
