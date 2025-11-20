# SegmentPro API - Official Documentation

> **SAM 3 Powered Object Segmentation SaaS**
> Version 1.0.0 | Last Updated: November 2025

## Overview

SegmentPro API is a simple, monetizable SaaS product built on Meta's SAM 3 (Segment Anything Model 3). It provides REST API endpoints for text-prompted object segmentation in images and videos.

### Key Features

- **Text-Prompted Segmentation**: Describe what you want to segment using natural language
- **Multi-Modal Prompts**: Support for text, bounding boxes, points, and image exemplars
- **Video Tracking**: Track objects across video frames automatically
- **Batch Processing**: Process multiple images with the same prompt
- **Tiered Pricing**: Free to Enterprise tiers for different use cases

### Use Cases

- **E-commerce**: Extract products from lifestyle photos
- **Content Creation**: Remove backgrounds, isolate subjects
- **Data Annotation**: Auto-label training data for ML
- **Video Production**: Track and mask objects in videos

---

## Quick Start

### 1. Installation

```bash
# Clone and setup
cd segmentpro_api
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the server
uvicorn app.main:app --reload
```

### 2. Get API Key

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "yourpassword"}'

# Get your API key from response or dashboard
```

### 3. Make Your First Request

```bash
curl -X POST http://localhost:8000/segment/image/upload \
  -H "X-API-Key: sp_your_api_key" \
  -F "file=@product.jpg" \
  -F "prompt=product"
```

---

## API Reference

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API documentation.

### Core Endpoints

| Endpoint | Method | Description | Tier |
|----------|--------|-------------|------|
| `/segment/image` | POST | Segment image from URL | Free+ |
| `/segment/image/upload` | POST | Segment uploaded image | Free+ |
| `/segment/video` | POST | Segment and track in video | Starter+ |
| `/segment/batch` | POST | Batch process images | Pro+ |
| `/jobs/{id}` | GET | Check job status | All |

### Authentication

Two methods supported:
- **API Key**: `X-API-Key: sp_your_key` header
- **JWT Token**: `Authorization: Bearer <token>` header

---

## Pricing Tiers

| Tier | Price | API Calls/Month | Features |
|------|-------|-----------------|----------|
| **Free** | $0 | 100 | Image segmentation, 5MB max |
| **Starter** | $19/mo | 1,000 | + Video segmentation, 10MB max |
| **Pro** | $49/mo | 5,000 | + Batch processing, webhooks |
| **Business** | $149/mo | 25,000 | + Dedicated GPU, SLA |
| **Enterprise** | Custom | Unlimited | + On-premise, 24/7 support |

---

## Architecture

```
segmentpro_api/
├── app/
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration & pricing
│   ├── models.py        # Database models
│   ├── schemas.py       # Pydantic schemas
│   ├── auth.py          # Authentication
│   ├── database.py      # DB connection
│   ├── sam3_service.py  # SAM 3 model wrapper
│   └── usage.py         # Usage tracking
├── docs/                # Documentation
├── tests/               # Test suite
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Technology Stack

- **Framework**: FastAPI (Python 3.12+)
- **Model**: Meta SAM 3 (848M parameters)
- **Database**: PostgreSQL / SQLite
- **Cache**: Redis
- **Payments**: Stripe
- **Deployment**: Docker, NVIDIA GPU

---

## Development Guide

### Running Locally

```bash
# Development server with hot reload
uvicorn app.main:app --reload --port 8000

# API docs available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black app/ tests/
```

### Database Migrations

```bash
alembic upgrade head
```

---

## Deployment

### Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### Manual Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- AWS EC2 with GPU setup
- GCP Compute Engine deployment
- Kubernetes with GPU nodes
- SSL/TLS configuration

---

## SAM 3 Model Setup

### Download Checkpoints

1. Request access at [SAM 3 Hugging Face](https://huggingface.co/facebook/sam3)
2. Accept the license agreement
3. Authenticate locally:

```bash
huggingface-cli login
```

4. Download checkpoints:

```bash
mkdir -p models/sam3
cd models/sam3
# Checkpoints will be downloaded automatically on first run
```

### Available Checkpoints

| Checkpoint | Parameters | Performance | GPU Memory |
|------------|------------|-------------|------------|
| sam3_tiny | 40M | Fast | 4GB |
| sam3_small | 98M | Balanced | 8GB |
| sam3_base_plus | 308M | Good | 12GB |
| sam3_large | 848M | Best | 24GB |

---

## Monetization Strategy

### Revenue Streams

1. **Subscription Tiers**: Monthly recurring revenue
2. **Overage Charges**: $0.01 per extra API call
3. **Enterprise Contracts**: Annual commitments
4. **Fine-tuning Services**: Custom model training

### Marketing Channels

- Developer communities (HackerNews, Reddit, Twitter)
- SEO for "image segmentation API", "background removal API"
- Affiliate program with design tools
- Content marketing (tutorials, case studies)

### Key Metrics to Track

- Monthly Recurring Revenue (MRR)
- Customer Acquisition Cost (CAC)
- Lifetime Value (LTV)
- Churn Rate
- API Usage per Customer

---

## Maintenance Guide

### Daily Tasks

- Monitor error logs
- Check API response times
- Review usage patterns

### Weekly Tasks

- Backup database
- Review security alerts
- Update dependencies if needed

### Monthly Tasks

- Analyze revenue metrics
- Review customer feedback
- Plan feature updates

---

## Troubleshooting

### Common Issues

**Model not loading**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify checkpoint path
ls -la models/sam3/
```

**Out of memory**
```bash
# Use smaller checkpoint
SAM3_CHECKPOINT=sam3_small

# Or enable CPU offloading
DEVICE=cpu
```

**Rate limiting**
- Upgrade tier or wait for monthly reset
- Check `/usage` endpoint for current usage

---

## Additional Documentation

- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - Complete API documentation
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guides
- [docs/INTEGRATION.md](docs/INTEGRATION.md) - SDK and integration examples
- [docs/SECURITY.md](docs/SECURITY.md) - Security best practices

---

## Support

- **Email**: support@segmentpro.io
- **GitHub Issues**: For bugs and feature requests
- **Discord**: Community support

---

## License

This project uses Meta's SAM 3 model under its license terms. The API wrapper code is MIT licensed.

---

## Changelog

### v1.0.0 (November 2025)
- Initial release
- SAM 3 integration
- Image and video segmentation
- Tiered pricing model
- Usage tracking and billing
