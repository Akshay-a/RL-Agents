# SegmentPro API - Integration Guide

Examples and best practices for integrating SegmentPro API into your applications.

---

## Python SDK

### Installation

```bash
pip install segmentpro-sdk
```

### Basic Usage

```python
from segmentpro import SegmentProClient

# Initialize client
client = SegmentProClient(api_key="sp_your_api_key")

# Segment local image
result = client.segment_image(
    image_path="product.jpg",
    prompt="product"
)

# Save masks
for i, mask in enumerate(result.masks):
    mask.save(f"mask_{i}.png")

# Get bounding boxes
for box, score in zip(result.boxes, result.scores):
    print(f"Box: {box}, Confidence: {score:.2f}")
```

### Segment from URL

```python
result = client.segment_image_url(
    image_url="https://example.com/image.jpg",
    prompt="person wearing red"
)
```

### Video Segmentation

```python
# Requires Starter tier+
result = client.segment_video(
    video_path="input.mp4",
    prompt="person",
    output_path="output.mp4"
)

print(f"Tracked {result.objects_tracked} objects across {result.frames_processed} frames")
```

### Batch Processing

```python
# Requires Pro tier+
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

job = client.segment_batch(
    images=images,
    prompt="product"
)

# Wait for completion
result = job.wait()

for i, item in enumerate(result.items):
    item.mask.save(f"batch_mask_{i}.png")
```

### Error Handling

```python
from segmentpro.exceptions import (
    AuthenticationError,
    RateLimitError,
    TierRestrictionError
)

try:
    result = client.segment_video(video_path="video.mp4", prompt="car")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except TierRestrictionError:
    print("Upgrade to Starter tier for video segmentation")
```

---

## JavaScript/Node.js SDK

### Installation

```bash
npm install segmentpro-sdk
```

### Basic Usage

```javascript
const SegmentPro = require('segmentpro-sdk');

const client = new SegmentPro({ apiKey: 'sp_your_api_key' });

// Segment image
async function segmentProduct() {
  const result = await client.segmentImage({
    imagePath: 'product.jpg',
    prompt: 'product'
  });

  console.log(`Found ${result.objectsDetected} objects`);

  // Save mask as PNG
  await result.masks[0].save('mask.png');

  return result;
}

segmentProduct().catch(console.error);
```

### Browser Usage

```javascript
import { SegmentProClient } from 'segmentpro-sdk/browser';

const client = new SegmentProClient({ apiKey: 'sp_your_key' });

// From file input
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];

  const result = await client.segmentImage({
    file: file,
    prompt: document.getElementById('prompt').value
  });

  // Display mask on canvas
  const canvas = document.getElementById('maskCanvas');
  result.masks[0].drawToCanvas(canvas);
});
```

---

## cURL Examples

### Image Segmentation

```bash
# With file upload
curl -X POST https://api.segmentpro.io/v1/segment/image/upload \
  -H "X-API-Key: sp_your_key" \
  -F "file=@product.jpg" \
  -F "prompt=main product"

# With URL
curl -X POST https://api.segmentpro.io/v1/segment/image \
  -H "X-API-Key: sp_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "prompt": "red car"
  }'
```

### Video Segmentation

```bash
curl -X POST https://api.segmentpro.io/v1/segment/video \
  -H "X-API-Key: sp_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "prompt": "person",
    "track_objects": true
  }'
```

### Check Usage

```bash
curl -X GET https://api.segmentpro.io/v1/usage \
  -H "X-API-Key: sp_your_key"
```

---

## Common Integration Patterns

### E-commerce Product Extraction

```python
from segmentpro import SegmentProClient
from PIL import Image

client = SegmentProClient(api_key="sp_your_key")

def extract_product(image_path: str, output_path: str):
    """Extract product from lifestyle photo with transparent background"""

    # Get segmentation mask
    result = client.segment_image(
        image_path=image_path,
        prompt="main product"
    )

    if not result.masks:
        raise ValueError("No product found in image")

    # Apply mask to original image
    original = Image.open(image_path).convert("RGBA")
    mask = result.masks[0].to_pil()

    # Create transparent background
    output = Image.new("RGBA", original.size, (0, 0, 0, 0))
    output.paste(original, mask=mask)

    # Crop to bounding box
    box = result.boxes[0]
    output = output.crop(box)

    output.save(output_path, "PNG")
    return output_path

# Usage
extract_product("lifestyle_photo.jpg", "product_only.png")
```

### Background Replacement

```python
def replace_background(foreground_path: str, background_path: str, output_path: str):
    """Replace background behind main subject"""

    # Segment foreground
    result = client.segment_image(
        image_path=foreground_path,
        prompt="person"  # or main subject
    )

    foreground = Image.open(foreground_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    mask = result.masks[0].to_pil()

    # Resize background to match
    background = background.resize(foreground.size)

    # Composite
    output = Image.composite(foreground, background, mask)
    output.save(output_path)

    return output_path
```

### Video Object Tracking

```python
def track_object_in_video(video_path: str, prompt: str):
    """Track and highlight object throughout video"""

    result = client.segment_video(
        video_path=video_path,
        prompt=prompt,
        track_objects=True
    )

    # Get tracking data
    tracking = result.tracking_data

    # Process each frame
    for frame_data in tracking:
        frame_idx = frame_data["frame"]
        box = frame_data["box"]
        score = frame_data["score"]

        # Draw bounding box, apply effects, etc.
        print(f"Frame {frame_idx}: {box} ({score:.2f})")

    return result
```

### Webhook Integration

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def handle_webhook():
    """Handle SegmentPro webhook events"""

    event = request.json
    event_type = event["event"]

    if event_type == "job.completed":
        job_id = event["data"]["job_id"]
        objects = event["data"]["objects_detected"]

        # Download results, update database, notify user, etc.
        print(f"Job {job_id} completed: {objects} objects found")

    elif event_type == "job.failed":
        job_id = event["data"]["job_id"]
        error = event["data"]["error"]

        # Handle failure
        print(f"Job {job_id} failed: {error}")

    return {"status": "ok"}
```

---

## Framework Integrations

### Django

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from segmentpro import SegmentProClient

client = SegmentProClient(api_key=settings.SEGMENTPRO_API_KEY)

@require_POST
def segment_upload(request):
    """Handle image upload and segmentation"""

    image = request.FILES.get("image")
    prompt = request.POST.get("prompt", "main object")

    if not image:
        return JsonResponse({"error": "No image provided"}, status=400)

    # Save temporarily
    temp_path = f"/tmp/{image.name}"
    with open(temp_path, "wb") as f:
        for chunk in image.chunks():
            f.write(chunk)

    # Segment
    result = client.segment_image(
        image_path=temp_path,
        prompt=prompt
    )

    return JsonResponse({
        "job_id": result.job_id,
        "objects_detected": result.objects_detected,
        "boxes": result.boxes
    })
```

### FastAPI

```python
from fastapi import FastAPI, UploadFile, File
from segmentpro import SegmentProClient

app = FastAPI()
client = SegmentProClient(api_key="sp_your_key")

@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = "main object"
):
    # Save file
    contents = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Segment
    result = client.segment_image(
        image_path=temp_path,
        prompt=prompt
    )

    return {
        "job_id": result.job_id,
        "objects": result.objects_detected
    }
```

### React

```jsx
import { useState } from 'react';

function ImageSegmenter() {
  const [mask, setMask] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', 'main subject');

    setLoading(true);

    try {
      const response = await fetch('https://api.segmentpro.io/v1/segment/image/upload', {
        method: 'POST',
        headers: {
          'X-API-Key': process.env.REACT_APP_SEGMENTPRO_KEY
        },
        body: formData
      });

      const result = await response.json();
      setMask(result.masks[0]);
    } catch (error) {
      console.error('Segmentation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} accept="image/*" />
      {loading && <p>Processing...</p>}
      {mask && <img src={`data:image/png;base64,${mask}`} alt="Mask" />}
    </div>
  );
}
```

---

## Best Practices

### 1. Prompt Engineering

```python
# Bad - too vague
result = client.segment_image(image, prompt="thing")

# Good - specific
result = client.segment_image(image, prompt="red sneaker")

# Better - with context
result = client.segment_image(image, prompt="red sneaker on white background")
```

### 2. Error Handling & Retries

```python
import time
from segmentpro.exceptions import RateLimitError, ServerError

def segment_with_retry(image_path, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.segment_image(image_path, prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(e.retry_after or 2 ** attempt)
            else:
                raise
        except ServerError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

### 3. Caching Results

```python
import hashlib
import json
from pathlib import Path

def get_cached_segment(image_path, prompt, cache_dir="./cache"):
    # Create cache key
    with open(image_path, "rb") as f:
        image_hash = hashlib.md5(f.read()).hexdigest()
    cache_key = hashlib.md5(f"{image_hash}{prompt}".encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.json"

    # Check cache
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Segment and cache
    result = client.segment_image(image_path, prompt)
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(result.to_dict()))

    return result
```

### 4. Monitor Usage

```python
def check_usage_before_batch(image_count):
    """Check if we have enough API calls remaining"""
    usage = client.get_usage()

    if usage.api_calls_remaining < image_count:
        raise Exception(
            f"Not enough API calls. Need {image_count}, have {usage.api_calls_remaining}"
        )

    return True
```
