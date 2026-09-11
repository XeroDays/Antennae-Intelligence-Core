"""Custom hand gesture detection from MediaPipe landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Tip and PIP landmark indices for each finger.
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_PIPS = (3, 6, 10, 14, 18)

# Wrist and middle MCP used for hand orientation.
WRIST = 0
MIDDLE_MCP = 9
THUMB_TIP = 4
INDEX_TIP = 8

DEFAULT_PINCH_THRESHOLD = 0.09


@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def count_extended(self) -> int:
        return sum(self.as_tuple())


class GestureDetector:
    """Detect finger states and named gestures from normalized landmarks."""

    def __init__(self, pinch_threshold: float = DEFAULT_PINCH_THRESHOLD) -> None:
        self.pinch_threshold = pinch_threshold

    def set_pinch_threshold(self, value: float) -> None:
        """Update the max thumb-index distance that counts as a pinch."""
        self.pinch_threshold = max(0.01, min(0.30, value))

    def get_finger_states(
        self,
        landmarks: Sequence,
        handedness: str,
    ) -> FingerState:
        """Return whether each finger is extended."""
        is_right_hand = handedness.lower().startswith("r")

        thumb_extended = self._is_thumb_extended(
            landmarks, is_right_hand=is_right_hand
        )
        index_extended = self._is_finger_extended(landmarks, 1)
        middle_extended = self._is_finger_extended(landmarks, 2)
        ring_extended = self._is_finger_extended(landmarks, 3)
        pinky_extended = self._is_finger_extended(landmarks, 4)

        return FingerState(
            thumb=thumb_extended,
            index=index_extended,
            middle=middle_extended,
            ring=ring_extended,
            pinky=pinky_extended,
        )

    def detect_gesture(
        self,
        landmarks: Sequence,
        handedness: str,
    ) -> tuple[str, FingerState]:
        """Return gesture label and finger state for one hand."""
        fingers = self.get_finger_states(landmarks, handedness)

        if self._is_pinch(landmarks):
            return "Pinch", fingers

        extended = fingers.as_tuple()

        if all(extended):
            return "Open Palm", fingers

        if not any(extended):
            return "Fist", fingers

        if extended[1] and not extended[2] and not extended[3] and not extended[4]:
            return "Point", fingers

        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            return "Peace", fingers

        if extended[0] and not extended[1] and not extended[2] and not extended[3] and not extended[4]:
            return "Thumbs Up", fingers

        if extended[1] and extended[4] and not extended[2] and not extended[3]:
            return "Rock", fingers

        return "Unknown", fingers

    def _is_finger_extended(self, landmarks: Sequence, finger_index: int) -> bool:
        tip = landmarks[FINGER_TIPS[finger_index]]
        pip = landmarks[FINGER_PIPS[finger_index]]
        return tip.y < pip.y

    def _is_thumb_extended(self, landmarks: Sequence, is_right_hand: bool) -> bool:
        tip = landmarks[THUMB_TIP]
        pip = landmarks[FINGER_PIPS[0]]
        wrist = landmarks[WRIST]

        # Thumb moves mostly on the x-axis; compare against wrist for orientation.
        if is_right_hand:
            return tip.x < pip.x and tip.x < wrist.x
        return tip.x > pip.x and tip.x > wrist.x

    def _is_pinch(self, landmarks: Sequence) -> bool:
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        distance = math.dist((thumb.x, thumb.y), (index.x, index.y))
        return distance < self.pinch_threshold
