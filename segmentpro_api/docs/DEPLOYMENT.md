# SegmentPro API - Deployment Guide

Instructions for deploying SegmentPro API to various environments.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12.6+
- At least 24GB GPU memory for sam3_large (or use smaller checkpoints)
- Domain name and SSL certificate (for production)

---

## Quick Deploy (Docker Compose)

### 1. Clone and Configure

```bash
cd segmentpro_api
cp .env.example .env
# Edit .env with your production values
```

### 2. Download SAM 3 Checkpoints

```bash
# Authenticate with Hugging Face
huggingface-cli login

# Create models directory
mkdir -p models/sam3

# Model will download on first run, or pre-download:
python -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"
```

### 3. Start Services

```bash
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### 4. Verify Deployment

```bash
curl http://localhost:8000/health
```

---

## AWS EC2 Deployment

### 1. Launch Instance

- **AMI**: Deep Learning AMI GPU PyTorch 2.7 (Ubuntu 24.04)
- **Instance Type**: g5.xlarge (or g5.2xlarge for better performance)
- **Storage**: 100GB gp3
- **Security Group**: Allow ports 22, 80, 443

### 2. Connect and Setup

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repository
git clone https://github.com/youruser/segmentpro_api.git
cd segmentpro_api

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker

# Setup NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env
```

**Production .env:**
```bash
DEBUG=false
SECRET_KEY=your-secure-random-key
DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/segmentpro
REDIS_URL=redis://elasticache-endpoint:6379
STRIPE_SECRET_KEY=sk_live_...
```

### 4. Start Application

```bash
docker-compose up -d
```

### 5. Setup Nginx (Optional but Recommended)

```bash
sudo apt install -y nginx certbot python3-certbot-nginx

# Configure nginx
sudo nano /etc/nginx/sites-available/segmentpro
```

```nginx
server {
    listen 80;
    server_name api.segmentpro.io;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # For file uploads
        client_max_body_size 100M;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/segmentpro /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d api.segmentpro.io
```

---

## GCP Compute Engine

### 1. Create VM

```bash
gcloud compute instances create segmentpro-api \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

### 2. Setup (Same as AWS)

Follow steps 2-5 from AWS deployment.

---

## Kubernetes Deployment

### 1. Create Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: segmentpro-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: segmentpro-api
  template:
    metadata:
      labels:
        app: segmentpro-api
    spec:
      containers:
      - name: api
        image: your-registry/segmentpro-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            memory: "16Gi"
        envFrom:
        - secretRef:
            name: segmentpro-secrets
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

### 2. Create Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: segmentpro-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: segmentpro-api
```

### 3. Deploy

```bash
kubectl apply -f k8s/
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | - | JWT signing key |
| `DATABASE_URL` | Yes | sqlite | Database connection |
| `REDIS_URL` | No | localhost | Redis for caching |
| `STRIPE_SECRET_KEY` | No | - | Stripe API key |
| `SAM3_CHECKPOINT` | No | sam3_large | Model size |
| `DEVICE` | No | cuda | cuda or cpu |

---

## Scaling

### Horizontal Scaling

For high traffic, deploy multiple API instances behind a load balancer:

```yaml
# docker-compose.scale.yml
services:
  api:
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Vertical Scaling

Choose checkpoint based on available GPU memory:

| Checkpoint | GPU Memory | Throughput |
|------------|------------|------------|
| sam3_tiny | 4GB | ~50 img/s |
| sam3_small | 8GB | ~30 img/s |
| sam3_base_plus | 12GB | ~20 img/s |
| sam3_large | 24GB | ~10 img/s |

---

## Monitoring

### Health Checks

The API provides a `/health` endpoint for load balancer health checks:

```bash
curl http://localhost:8000/health
```

### Logging

Logs are structured JSON for easy parsing:

```bash
docker-compose logs -f api | jq .
```

### Metrics

Consider adding Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, generate_latest

REQUESTS = Counter('requests_total', 'Total requests')
LATENCY = Histogram('request_latency_seconds', 'Request latency')
```

---

## Backup Strategy

### Database

```bash
# PostgreSQL backup
pg_dump -h db-host -U segmentpro segmentpro > backup.sql

# Restore
psql -h db-host -U segmentpro segmentpro < backup.sql
```

### Automated Backups (AWS RDS)

Enable automated backups in RDS console with 7-30 day retention.

---

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Use HTTPS only
- [ ] Enable database SSL
- [ ] Set up firewall rules
- [ ] Configure rate limiting
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Rotate API keys periodically

---

## Cost Optimization

### GPU Costs

| Cloud | Instance | Hourly Cost | Monthly |
|-------|----------|-------------|---------|
| AWS | g5.xlarge | $1.006 | ~$724 |
| GCP | n1+T4 | $0.95 | ~$684 |
| Azure | NC4as_T4_v3 | $0.526 | ~$379 |

### Tips

1. Use spot/preemptible instances for dev/test
2. Auto-scale based on demand
3. Use smaller checkpoints during off-peak
4. Cache frequent segmentation results
