# SegmentPro API - Security Guide

Security best practices for deploying and using SegmentPro API.

---

## API Key Security

### Protecting Your API Keys

**DO:**
- Store API keys in environment variables
- Use different keys for development and production
- Rotate keys periodically (every 90 days recommended)
- Use server-side requests only (never expose in client code)

**DON'T:**
- Commit API keys to version control
- Share keys in chat/email
- Use production keys in development
- Expose keys in browser JavaScript

### Key Rotation

```bash
# Create new key
curl -X POST https://api.segmentpro.io/v1/api-keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Production v2"}'

# Update your application with new key
# Then delete old key
curl -X DELETE https://api.segmentpro.io/v1/api-keys/OLD_KEY_ID \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Securing Keys in Applications

**Python/Django:**
```python
# settings.py
import os
SEGMENTPRO_API_KEY = os.environ.get('SEGMENTPRO_API_KEY')
```

**Node.js:**
```javascript
// Use dotenv
require('dotenv').config();
const apiKey = process.env.SEGMENTPRO_API_KEY;
```

**Docker:**
```yaml
# docker-compose.yml
services:
  app:
    environment:
      - SEGMENTPRO_API_KEY=${SEGMENTPRO_API_KEY}
```

---

## Authentication Best Practices

### Password Requirements

- Minimum 8 characters
- Mix of uppercase, lowercase, numbers
- No common passwords
- Consider implementing:
  - Password strength meter
  - Breach database checking
  - Rate limiting on login attempts

### JWT Token Security

```python
# Short-lived tokens (30 minutes default)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Use secure algorithm
ALGORITHM=HS256

# Strong secret key
SECRET_KEY=your-256-bit-secret-key
```

### Token Storage (Client-side)

```javascript
// Good: HttpOnly cookie (server sets)
// Bad: localStorage (XSS vulnerable)

// If using localStorage, implement token refresh
async function authenticatedRequest(url, options) {
  const token = localStorage.getItem('access_token');

  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });

  if (response.status === 401) {
    // Token expired, refresh or redirect to login
    await refreshToken();
    return authenticatedRequest(url, options);
  }

  return response;
}
```

---

## Data Protection

### Input Validation

All inputs are validated using Pydantic schemas:

```python
class SegmentImageRequest(BaseModel):
    image_url: Optional[str] = None
    prompt: str = Field(..., min_length=1, max_length=500)
    prompt_type: PromptType = PromptType.TEXT
```

### File Upload Security

```python
# Validate file type
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

async def validate_upload(file: UploadFile):
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")

    # Check content type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")

    # Check file size (handled by tier limits)
    pass
```

### Preventing Injection Attacks

- SQL injection: Using SQLAlchemy ORM with parameterized queries
- Command injection: No shell execution with user input
- XSS: Proper content-type headers, no HTML rendering of user input

---

## Network Security

### HTTPS Only

Always use HTTPS in production:

```nginx
server {
    listen 80;
    server_name api.segmentpro.io;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.segmentpro.io;

    ssl_certificate /etc/letsencrypt/live/api.segmentpro.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.segmentpro.io/privkey.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
}
```

### CORS Configuration

```python
# Production: Restrict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourapp.com",
        "https://www.yourapp.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)
```

### Rate Limiting

Implemented to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/segment/image")
@limiter.limit("60/minute")  # Adjust per tier
async def segment_image(request: Request):
    pass
```

---

## Infrastructure Security

### Database Security

```python
# Use SSL for database connections
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Rotate database credentials
# Use IAM database authentication (AWS RDS)
```

### Redis Security

```bash
# Enable authentication
REDIS_URL=redis://:password@localhost:6379

# Disable dangerous commands
# In redis.conf:
rename-command FLUSHALL ""
rename-command CONFIG ""
```

### Container Security

```dockerfile
# Don't run as root
FROM python:3.12-slim
RUN useradd -m appuser
USER appuser

# Use specific versions
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Scan for vulnerabilities
# docker scan your-image:tag
```

### Secrets Management

For production, use a secrets manager:

**AWS Secrets Manager:**
```python
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Usage
stripe_key = get_secret('prod/segmentpro/stripe')
```

---

## Monitoring & Logging

### Security Logging

```python
import structlog

logger = structlog.get_logger()

# Log authentication events
logger.info("user_login", user_id=user.id, ip=request.client.host)
logger.warning("failed_login", email=email, ip=request.client.host)

# Log API key usage
logger.info("api_key_used", key_id=api_key.id, endpoint=endpoint)

# Log suspicious activity
logger.warning("rate_limit_exceeded", user_id=user.id, endpoint=endpoint)
```

### Alerting

Set up alerts for:
- Multiple failed login attempts
- Rate limit exceeded
- Unusual usage patterns
- Error rate spikes

---

## Compliance

### Data Retention

```python
# Delete old usage records (example: 90 days)
def cleanup_old_records():
    cutoff = datetime.utcnow() - timedelta(days=90)
    db.query(UsageRecord).filter(UsageRecord.created_at < cutoff).delete()
```

### User Data Requests

Support GDPR/CCPA requests:

```python
# Export user data
def export_user_data(user_id):
    user = db.query(User).get(user_id)
    usage = db.query(UsageRecord).filter_by(user_id=user_id).all()

    return {
        "user": user.to_dict(),
        "usage_records": [r.to_dict() for r in usage]
    }

# Delete user data
def delete_user_data(user_id):
    db.query(UsageRecord).filter_by(user_id=user_id).delete()
    db.query(APIKey).filter_by(user_id=user_id).delete()
    db.query(User).filter_by(id=user_id).delete()
    db.commit()
```

---

## Security Checklist

### Before Launch

- [ ] Change all default passwords and keys
- [ ] Enable HTTPS with valid certificate
- [ ] Configure firewall rules
- [ ] Set up database backups
- [ ] Enable logging and monitoring
- [ ] Review CORS settings
- [ ] Test rate limiting
- [ ] Scan dependencies for vulnerabilities

### Ongoing

- [ ] Rotate API keys quarterly
- [ ] Update dependencies monthly
- [ ] Review access logs weekly
- [ ] Test backup restoration quarterly
- [ ] Security audit annually

---

## Reporting Vulnerabilities

If you discover a security vulnerability:

1. **Do not** disclose publicly
2. Email security@segmentpro.io with details
3. Include steps to reproduce
4. We'll respond within 48 hours
5. We'll credit you in our security acknowledgments

---

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
