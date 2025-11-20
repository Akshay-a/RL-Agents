# SegmentPro API

**SAM 3 Powered Object Segmentation SaaS**

A simple, monetizable API service for text-prompted object segmentation using Meta's SAM 3 model.

## Quick Start

```bash
# Setup
cd segmentpro_api
pip install -r requirements.txt
cp .env.example .env

# Run
uvicorn app.main:app --reload
```

## Documentation

See [claude.md](claude.md) for complete documentation, including:
- API Reference
- Deployment Guide
- Integration Examples
- Pricing Tiers

## Additional Docs

- [API Reference](docs/API_REFERENCE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Integration](docs/INTEGRATION.md)
- [Security](docs/SECURITY.md)

## License

MIT License (API code) | SAM 3 Model: Meta License
