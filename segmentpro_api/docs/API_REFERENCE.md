# SegmentPro API Reference

Complete API documentation for SegmentPro API v1.0.0

## Base URL

```
Production: https://api.segmentpro.io/v1
Development: http://localhost:8000
```

## Authentication

### API Key (Recommended)

Include your API key in the `X-API-Key` header:

```bash
curl -X POST https://api.segmentpro.io/v1/segment/image/upload \
  -H "X-API-Key: sp_your_api_key_here" \
  -F "file=@image.jpg" \
  -F "prompt=person"
```

### JWT Token

For dashboard access, use Bearer token:

```bash
curl -X GET https://api.segmentpro.io/v1/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

---

## Endpoints

### Authentication

#### Register User

```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "full_name": "John Doe",
  "company": "Acme Inc"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "company": "Acme Inc",
  "tier": "free",
  "is_active": true,
  "created_at": "2025-11-20T10:00:00Z"
}
```

#### Login

```http
POST /auth/login?email=user@example.com&password=securepassword123
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### Get Current User

```http
GET /auth/me
```

---

### API Keys

#### Create API Key

```http
POST /api-keys
```

**Request Body:**
```json
{
  "name": "Production App"
}
```

**Response:**
```json
{
  "id": 1,
  "key": "sp_xK9mN2pL4qR7sT1uV3wY5zA8bC0dE6fG",
  "name": "Production App",
  "is_active": true,
  "created_at": "2025-11-20T10:00:00Z",
  "last_used_at": null
}
```

#### List API Keys

```http
GET /api-keys
```

#### Delete API Key

```http
DELETE /api-keys/{key_id}
```

---

### Image Segmentation

#### Segment Image (URL)

```http
POST /segment/image
```

**Request Body:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "prompt": "red car",
  "prompt_type": "text",
  "output_format": "png",
  "return_mask": true,
  "return_box": true,
  "return_score": true,
  "multimask_output": false,
  "mask_threshold": 0.5
}
```

**Prompt Types:**
- `text` - Natural language description (default)
- `box` - Bounding box coordinates
- `point` - Click points with labels

**For box prompts:**
```json
{
  "prompt": "segment this box",
  "prompt_type": "box",
  "box": [100, 100, 300, 300]
}
```

**For point prompts:**
```json
{
  "prompt": "segment at these points",
  "prompt_type": "point",
  "points": [[150, 200], [180, 220]],
  "point_labels": [1, 1]
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "masks": ["base64_encoded_png..."],
  "boxes": [[120, 80, 350, 400]],
  "scores": [0.97],
  "objects_detected": 1,
  "processing_time_ms": 245,
  "created_at": "2025-11-20T10:00:00Z",
  "completed_at": "2025-11-20T10:00:00.245Z"
}
```

#### Segment Image (Upload)

```http
POST /segment/image/upload
Content-Type: multipart/form-data
```

**Form Fields:**
- `file` (required): Image file (JPEG, PNG, WebP)
- `prompt` (required): Text description
- `prompt_type` (optional): "text", "box", "point"

**Example:**
```bash
curl -X POST https://api.segmentpro.io/v1/segment/image/upload \
  -H "X-API-Key: sp_your_key" \
  -F "file=@product.jpg" \
  -F "prompt=product on white background"
```

---

### Video Segmentation

#### Segment Video

**Requires: Starter tier or higher**

```http
POST /segment/video
```

**Request Body:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "prompt": "person in red shirt",
  "prompt_type": "text",
  "start_frame": 0,
  "end_frame": 300,
  "track_objects": true,
  "output_format": "mp4"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "results": {
    "frames_processed": 300,
    "objects_tracked": 1,
    "tracking_data": [
      {"frame": 0, "box": [100, 100, 200, 300], "score": 0.95},
      {"frame": 1, "box": [102, 98, 202, 298], "score": 0.94}
    ]
  },
  "processing_time_ms": 5200,
  "created_at": "2025-11-20T10:00:00Z",
  "completed_at": "2025-11-20T10:00:05.200Z"
}
```

---

### Batch Processing

#### Batch Segment Images

**Requires: Pro tier or higher**

```http
POST /segment/batch
```

**Request Body:**
```json
{
  "images": [
    "https://example.com/img1.jpg",
    "https://example.com/img2.jpg",
    "https://example.com/img3.jpg"
  ],
  "prompt": "main product",
  "prompt_type": "text",
  "output_format": "png"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "pending",
  "progress": 0.0
}
```

---

### Jobs

#### Get Job Status

```http
GET /jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "processing",
  "progress": 0.67,
  "result": null
}
```

**Status Values:**
- `pending` - Job queued
- `processing` - Currently running
- `completed` - Finished successfully
- `failed` - Error occurred

---

### Usage & Billing

#### Get Usage Statistics

```http
GET /usage
```

**Response:**
```json
{
  "billing_period": "2025-11",
  "api_calls_used": 847,
  "api_calls_limit": 5000,
  "api_calls_remaining": 4153,
  "image_segmentations": 800,
  "video_segmentations": 40,
  "batch_jobs": 7,
  "current_tier": "pro",
  "estimated_cost": 49.00
}
```

#### Get Pricing Tiers

```http
GET /pricing
```

#### Update Subscription

```http
POST /subscription
```

**Request Body:**
```json
{
  "tier": "pro",
  "payment_method_id": "pm_1234567890"
}
```

---

### Webhooks

#### Create Webhook

**Requires: Pro tier or higher**

```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://yourapp.com/webhook",
  "events": ["job.completed", "job.failed"]
}
```

**Available Events:**
- `job.completed` - Segmentation job finished
- `job.failed` - Job encountered error
- `usage.limit_reached` - Monthly limit hit

**Webhook Payload:**
```json
{
  "event": "job.completed",
  "timestamp": "2025-11-20T10:00:05Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "objects_detected": 3
  }
}
```

---

### System

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true
}
```

#### Model Info

```http
GET /model/info
```

**Response:**
```json
{
  "model": "SAM 3",
  "checkpoint": "sam3_large",
  "device": "cuda",
  "loaded": true,
  "parameters": "848M",
  "capabilities": [
    "text_prompts",
    "box_prompts",
    "point_prompts",
    "video_tracking",
    "concept_segmentation"
  ]
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing auth |
| 403 | Forbidden - Insufficient permissions/tier |
| 404 | Not Found |
| 413 | Payload Too Large - File exceeds tier limit |
| 429 | Too Many Requests - Rate/usage limit exceeded |
| 500 | Internal Server Error |

### Common Errors

**401 - Invalid authentication**
```json
{
  "detail": "Invalid authentication credentials"
}
```

**403 - Tier restriction**
```json
{
  "detail": "Video segmentation requires Starter tier or higher"
}
```

**429 - Usage limit reached**
```json
{
  "detail": "Monthly API limit reached (5000 calls). Please upgrade your plan."
}
```

---

## Rate Limits

| Tier | Requests/Minute | Concurrent Jobs |
|------|-----------------|-----------------|
| Free | 10 | 1 |
| Starter | 60 | 3 |
| Pro | 120 | 10 |
| Business | 300 | 25 |
| Enterprise | Custom | Custom |

---

## SDKs

### Python

```python
from segmentpro import SegmentProClient

client = SegmentProClient(api_key="sp_your_key")

# Segment image
result = client.segment_image(
    image_path="product.jpg",
    prompt="main product"
)

print(f"Found {result.objects_detected} objects")
for mask in result.masks:
    mask.save(f"mask_{mask.index}.png")
```

### JavaScript

```javascript
const SegmentPro = require('segmentpro');

const client = new SegmentPro({ apiKey: 'sp_your_key' });

const result = await client.segmentImage({
  imagePath: 'product.jpg',
  prompt: 'main product'
});

console.log(`Found ${result.objectsDetected} objects`);
```

---

## Best Practices

1. **Use specific prompts**: "red sports car" > "car"
2. **Cache results**: Store masks for repeated use
3. **Batch when possible**: Pro+ users should batch similar images
4. **Handle errors gracefully**: Implement retries with exponential backoff
5. **Monitor usage**: Check `/usage` regularly to avoid hitting limits
