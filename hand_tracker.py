"""MediaPipe hand tracking with landmark drawing and gesture labels."""

from __future__ import annotations

import math
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

from gesture_detector import FINGER_NAMES, INDEX_TIP, THUMB_TIP, GestureDetector

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

WRIST = 0
INTERLOCKED_WRIST_MIN = 0.03
INTERLOCKED_WRIST_THRESHOLD = 0.25
INTERLOCKED_OVERLAP_THRESHOLD = 0.40

DEFAULT_DETECTION_CONFIDENCE = 0.75
DEFAULT_PRESENCE_CONFIDENCE = 0.75
DEFAULT_TRACKING_CONFIDENCE = 0.60

NON_PINCH_FINGERS = ("middle", "ring", "pinky")


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
        self._detection_confidence = DEFAULT_DETECTION_CONFIDENCE
        self._presence_confidence = DEFAULT_PRESENCE_CONFIDENCE
        self._tracking_confidence = DEFAULT_TRACKING_CONFIDENCE
        self._landmarker = self._create_landmarker()

    def close(self) -> None:
        self._landmarker.close()

    def set_pinch_threshold(self, value: float) -> None:
        """Update pinch detection sensitivity."""
        self._detector.set_pinch_threshold(value)

    def reconfigure(
        self,
        detection_confidence: float,
        presence_confidence: float,
        tracking_confidence: float,
    ) -> None:
        """Rebuild the hand landmarker with new confidence thresholds."""
        self._detection_confidence = detection_confidence
        self._presence_confidence = presence_confidence
        self._tracking_confidence = tracking_confidence
        old_landmarker = self._landmarker
        self._landmarker = self._create_landmarker()
        old_landmarker.close()

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

    def detect_and_draw_interlocked(
        self,
        frame: np.ndarray,
        hand_results: list[HandResult],
    ) -> bool:
        """Detect clasped hands and draw an Interlocked label when both overlap."""
        if len(hand_results) < 2:
            return False

        frame_width = max(frame.shape[1], 1)
        wrist_a = hand_results[0].landmarks[WRIST]
        wrist_b = hand_results[1].landmarks[WRIST]
        wrist_dist = math.dist((wrist_a.x, wrist_a.y), (wrist_b.x, wrist_b.y)) / frame_width
        if wrist_dist < INTERLOCKED_WRIST_MIN or wrist_dist >= INTERLOCKED_WRIST_THRESHOLD:
            return False

        overlap_ratio = self._bounding_box_overlap_ratio(
            hand_results[0].landmarks,
            hand_results[1].landmarks,
        )
        if overlap_ratio <= INTERLOCKED_OVERLAP_THRESHOLD:
            return False

        self._draw_interlocked_label(
            frame,
            (wrist_a.x + wrist_b.x) // 2,
            (wrist_a.y + wrist_b.y) // 2,
        )
        return True

    def draw_volume_bridge(
        self,
        frame: np.ndarray,
        hand_results: list[HandResult],
    ) -> float | None:
        """Draw volume line between pinch midpoints when both hands are pinching."""
        if len(hand_results) < 2:
            return None

        if hand_results[0].gesture != "Pinch" or hand_results[1].gesture != "Pinch":
            return None

        for hand in (hand_results[0], hand_results[1]):
            if any(hand.finger_states.get(finger) for finger in NON_PINCH_FINGERS):
                return None

        point_a = self._pinch_midpoint(hand_results[0].landmarks)
        point_b = self._pinch_midpoint(hand_results[1].landmarks)

        cv2.line(frame, point_a, point_b, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, point_a, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point_b, 8, (255, 255, 255), -1, cv2.LINE_AA)

        midpoint = (
            (point_a[0] + point_b[0]) // 2,
            (point_a[1] + point_b[1]) // 2,
        )
        cv2.circle(frame, midpoint, 5, (0, 200, 255), -1, cv2.LINE_AA)

        frame_width = max(frame.shape[1], 1)
        pixel_dist = math.dist(point_a, point_b)
        normalized_dist = pixel_dist / frame_width
        return max(0.0, min(1.0, normalized_dist))

    def _pinch_midpoint(self, landmarks: list[LandmarkPoint]) -> tuple[int, int]:
        """Return the midpoint between thumb tip and index tip for a pinch."""
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        return ((thumb.x + index.x) // 2, (thumb.y + index.y) // 2)

    def _bounding_box_overlap_ratio(
        self,
        landmarks_a: list[LandmarkPoint],
        landmarks_b: list[LandmarkPoint],
    ) -> float:
        """Return overlap area divided by the smaller hand bounding box area."""
        ax1, ay1, ax2, ay2 = self._landmark_bounding_box(landmarks_a)
        bx1, by1, bx2, by2 = self._landmark_bounding_box(landmarks_b)

        overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
        overlap_area = overlap_width * overlap_height

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return overlap_area / min(area_a, area_b)

    def _landmark_bounding_box(
        self,
        landmarks: list[LandmarkPoint],
    ) -> tuple[int, int, int, int]:
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    def _draw_interlocked_label(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> None:
        label = "Interlocked"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)

        x = center_x - text_width // 2
        y = center_y + text_height // 2
        cv2.rectangle(
            frame,
            (x - 12, y - text_height - 12),
            (x + text_width + 12, y + baseline + 12),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _create_landmarker(self) -> HandLandmarker:
        model_path = self._ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self._detection_confidence,
            min_hand_presence_confidence=self._presence_confidence,
            min_tracking_confidence=self._tracking_confidence,
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
