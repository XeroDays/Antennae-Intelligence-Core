"""MediaPipe face tracking for hand-on-face detection."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

FACE_BOX_PADDING = 0.10


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class FaceTracker:
    """Detect a face and return an expanded bounding box each frame."""

    def __init__(self) -> None:
        self._frame_timestamp_ms = 0
        self._landmarker = self._create_landmarker()

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame: np.ndarray) -> FaceBox | None:
        """Return the primary face bounding box for a BGR frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.face_landmarks:
            return None

        height, width = frame.shape[:2]
        face_landmarks = result.face_landmarks[0]
        xs = [landmark.x * width for landmark in face_landmarks]
        ys = [landmark.y * height for landmark in face_landmarks]

        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        pad_x = (x2 - x1) * FACE_BOX_PADDING
        pad_y = (y2 - y1) * FACE_BOX_PADDING

        return FaceBox(
            x1=max(0.0, x1 - pad_x),
            y1=max(0.0, y1 - pad_y),
            x2=min(float(width - 1), x2 + pad_x),
            y2=min(float(height - 1), y2 + pad_y),
        )

    def _create_landmarker(self) -> FaceLandmarker:
        model_path = self._ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
        )
        return FaceLandmarker.create_from_options(options)

    def _ensure_model(self) -> Path:
        if MODEL_PATH.exists():
            return MODEL_PATH

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH
