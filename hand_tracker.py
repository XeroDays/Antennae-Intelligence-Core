"""MediaPipe hand tracking with landmark drawing and gesture labels."""

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
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)

from gesture_detector import FINGER_NAMES, GestureDetector

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass
class LandmarkPoint:
    x: int
    y: int
    z: float


@dataclass
class HandResult:
    landmarks: list[LandmarkPoint]
    handedness: str
    gesture: str
    finger_states: dict[str, bool]


class HandTracker:
    """Detect up to two hands and overlay landmarks plus gesture labels."""

    def __init__(self) -> None:
        self._detector = GestureDetector()
        self._connections = HandLandmarksConnections.HAND_CONNECTIONS
        self._frame_timestamp_ms = 0
        self._landmarker = self._create_landmarker()

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame: np.ndarray) -> list[HandResult]:
        """Detect hands in a BGR frame and return landmark/gesture results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            return []

        height, width = frame.shape[:2]
        hand_results: list[HandResult] = []

        for index, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = self._to_pixel_landmarks(hand_landmarks, width, height)
            handedness = "Unknown"
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name or "Unknown"

            gesture, finger_state = self._detector.detect_gesture(
                hand_landmarks,
                handedness,
            )
            finger_states = {
                name: value
                for name, value in zip(FINGER_NAMES, finger_state.as_tuple())
            }

            hand_results.append(
                HandResult(
                    landmarks=landmarks,
                    handedness=handedness,
                    gesture=gesture,
                    finger_states=finger_states,
                )
            )

        return hand_results

    def draw(self, frame: np.ndarray, hand_results: list[HandResult]) -> np.ndarray:
        """Draw hand skeleton, pivot points, and gesture labels on the frame."""
        output = frame.copy()

        for index, hand in enumerate(hand_results):
            color = (
                (0, 255, 0)
                if hand.handedness.lower().startswith("l")
                else (255, 128, 0)
            )
            self._draw_landmarks(output, hand.landmarks, color)
            self._draw_gesture_label(output, hand, index)

        return output

    def _create_landmarker(self) -> HandLandmarker:
        model_path = self._ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)

    def _ensure_model(self) -> Path:
        if MODEL_PATH.exists():
            return MODEL_PATH

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH

    def _to_pixel_landmarks(
        self,
        hand_landmarks,
        width: int,
        height: int,
    ) -> list[LandmarkPoint]:
        return [
            LandmarkPoint(
                x=int(landmark.x * width),
                y=int(landmark.y * height),
                z=landmark.z,
            )
            for landmark in hand_landmarks
        ]

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[LandmarkPoint],
        color: tuple[int, int, int],
    ) -> None:
        for connection in self._connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            cv2.line(
                frame,
                (start.x, start.y),
                (end.x, end.y),
                color,
                2,
                cv2.LINE_AA,
            )

        for point in landmarks:
            cv2.circle(frame, (point.x, point.y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (point.x, point.y), 6, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_gesture_label(
        self,
        frame: np.ndarray,
        hand: HandResult,
        hand_index: int,
    ) -> None:
        if not hand.landmarks:
            return

        wrist = hand.landmarks[0]
        x = max(10, wrist.x - 40)
        y = max(30, wrist.y - 20 - (hand_index * 70))

        finger_summary = " ".join(
            f"{name[0].upper()}{'↑' if extended else '↓'}"
            for name, extended in hand.finger_states.items()
        )
        label = f"{hand.handedness}: {hand.gesture}  [{finger_summary}]"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        cv2.rectangle(
            frame,
            (x - 8, y - text_height - 10),
            (x + text_width + 8, y + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
