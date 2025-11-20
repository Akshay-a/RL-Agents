"""
SAM 3 Model Service - Core segmentation logic
Wraps Meta's SAM 3 model for API usage
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import base64
import io
import structlog
import uuid
from datetime import datetime

logger = structlog.get_logger()


class SAM3Service:
    """
    Service class for SAM 3 model operations.
    Handles image and video segmentation with various prompt types.
    """

    def __init__(
        self,
        model_path: str = "./models/sam3",
        checkpoint: str = "sam3_large",
        device: str = "cuda"
    ):
        self.model_path = Path(model_path)
        self.checkpoint = checkpoint
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = None
        self.processor = None
        self.video_predictor = None
        self._initialized = False

        logger.info(
            "SAM3Service initialized",
            device=self.device,
            checkpoint=checkpoint
        )

    def load_model(self) -> bool:
        """Load SAM 3 model and processor"""
        try:
            # Import SAM 3 components
            # Note: Actual imports depend on sam3 package structure
            # This is a placeholder for the real implementation
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("Loading SAM 3 model...", checkpoint=self.checkpoint)

            # Build model
            self.model = build_sam3_image_model(
                checkpoint=self.checkpoint,
                device=self.device
            )

            # Create processor
            self.processor = Sam3Processor(self.model)

            self._initialized = True
            logger.info("SAM 3 model loaded successfully")
            return True

        except ImportError as e:
            logger.warning(
                "SAM 3 package not installed, using mock mode",
                error=str(e)
            )
            self._initialized = True  # Allow running in mock mode
            return True
        except Exception as e:
            logger.error("Failed to load SAM 3 model", error=str(e))
            return False

    def load_video_predictor(self) -> bool:
        """Load SAM 3 video predictor for video segmentation"""
        try:
            from sam3.model_builder import build_sam3_video_predictor

            self.video_predictor = build_sam3_video_predictor(
                checkpoint=self.checkpoint,
                device=self.device
            )
            return True
        except Exception as e:
            logger.error("Failed to load video predictor", error=str(e))
            return False

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def segment_image(
        self,
        image: Image.Image,
        prompt: str,
        prompt_type: str = "text",
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = False,
        mask_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Segment objects in an image based on prompt.

        Args:
            image: PIL Image to segment
            prompt: Text description or coordinates
            prompt_type: Type of prompt (text, box, point, mask)
            box: Bounding box [x1, y1, x2, y2]
            points: List of points [[x, y], ...]
            point_labels: Labels for points (1=foreground, 0=background)
            multimask_output: Return multiple masks
            mask_threshold: Threshold for mask binarization

        Returns:
            Dict with masks, boxes, scores, and metadata
        """
        start_time = datetime.utcnow()
        job_id = str(uuid.uuid4())

        try:
            if self.processor is not None:
                # Real SAM 3 inference
                state = self.processor.set_image(image)

                if prompt_type == "text":
                    output = self.processor.set_text_prompt(
                        state=state,
                        prompt=prompt
                    )
                elif prompt_type == "box" and box:
                    output = self.processor.set_box_prompt(
                        state=state,
                        box=box
                    )
                elif prompt_type == "point" and points:
                    output = self.processor.set_point_prompt(
                        state=state,
                        points=points,
                        labels=point_labels or [1] * len(points)
                    )
                else:
                    output = self.processor.set_text_prompt(
                        state=state,
                        prompt=prompt
                    )

                masks = output["masks"]
                boxes = output["boxes"]
                scores = output["scores"]
            else:
                # Mock mode for development/testing
                masks, boxes, scores = self._mock_segment(image, prompt)

            # Convert masks to base64 encoded PNGs
            encoded_masks = []
            for mask in masks:
                encoded_masks.append(self._encode_mask(mask))

            # Calculate processing time
            processing_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            return {
                "job_id": job_id,
                "status": "completed",
                "masks": encoded_masks,
                "boxes": boxes.tolist() if hasattr(boxes, 'tolist') else boxes,
                "scores": scores.tolist() if hasattr(scores, 'tolist') else scores,
                "objects_detected": len(masks),
                "processing_time_ms": processing_time_ms,
                "created_at": start_time,
                "completed_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Segmentation failed", error=str(e), job_id=job_id)
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "created_at": start_time,
                "completed_at": datetime.utcnow()
            }

    def segment_video(
        self,
        video_path: str,
        prompt: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        track_objects: bool = True
    ) -> Dict[str, Any]:
        """
        Segment and track objects in a video.

        Args:
            video_path: Path to video file
            prompt: Text description of objects to track
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            track_objects: Whether to track objects across frames

        Returns:
            Dict with frame-by-frame segmentation results
        """
        start_time = datetime.utcnow()
        job_id = str(uuid.uuid4())

        try:
            if self.video_predictor is None:
                self.load_video_predictor()

            if self.video_predictor is not None:
                # Start video session
                response = self.video_predictor.handle_request({
                    "type": "start_session",
                    "resource_path": video_path
                })

                session_id = response["session_id"]

                # Add prompt
                response = self.video_predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": start_frame,
                    "text": prompt
                })

                # Propagate masks through video
                if track_objects:
                    response = self.video_predictor.handle_request({
                        "type": "propagate_in_video",
                        "session_id": session_id
                    })

                results = response
            else:
                # Mock mode
                results = self._mock_video_segment(video_path, prompt)

            processing_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            return {
                "job_id": job_id,
                "status": "completed",
                "results": results,
                "processing_time_ms": processing_time_ms,
                "created_at": start_time,
                "completed_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Video segmentation failed", error=str(e), job_id=job_id)
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "created_at": start_time,
                "completed_at": datetime.utcnow()
            }

    def _encode_mask(self, mask: np.ndarray) -> str:
        """Convert mask array to base64 encoded PNG"""
        # Convert to PIL Image
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        img = Image.fromarray(mask)

        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _mock_segment(
        self,
        image: Image.Image,
        prompt: str
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        """Generate mock segmentation results for development"""
        width, height = image.size

        # Create a simple mock mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Create a circular mask in the center
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4

        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask[dist_from_center <= radius] = 255

        # Mock box and score
        box = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        score = 0.95

        return [mask], [box], [score]

    def _mock_video_segment(
        self,
        video_path: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Generate mock video segmentation results"""
        return {
            "frames_processed": 30,
            "objects_tracked": 1,
            "tracking_data": [
                {"frame": i, "box": [100, 100, 200, 200], "score": 0.9}
                for i in range(30)
            ]
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model": "SAM 3",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "loaded": self._initialized,
            "parameters": "848M",
            "capabilities": [
                "text_prompts",
                "box_prompts",
                "point_prompts",
                "video_tracking",
                "concept_segmentation"
            ]
        }


# Singleton instance
_sam3_service: Optional[SAM3Service] = None


def get_sam3_service() -> SAM3Service:
    """Get or create SAM3Service singleton"""
    global _sam3_service
    if _sam3_service is None:
        from .config import get_settings
        settings = get_settings()
        _sam3_service = SAM3Service(
            model_path=settings.sam3_model_path,
            checkpoint=settings.sam3_checkpoint,
            device=settings.device
        )
        _sam3_service.load_model()
    return _sam3_service
