"""Windows screen brightness control with save and restore."""

from __future__ import annotations

import screen_brightness_control as sbc


class BrightnessController:
    """Save, dim, and restore primary display brightness."""

    def __init__(self) -> None:
        self._saved_brightness: int | None = None

    def get_brightness(self) -> int:
        """Return current primary display brightness as 0-100."""
        value = sbc.get_brightness(display=0)
        if isinstance(value, list):
            return int(value[0])
        return int(value)

    def set_brightness(self, level: int) -> None:
        """Set primary display brightness to a 0-100 level."""
        clamped = max(0, min(100, int(level)))
        sbc.set_brightness(clamped, display=0)

    def save_and_dim(self, target: int = 10) -> None:
        """Save current brightness and dim to the target level."""
        self._saved_brightness = self.get_brightness()
        self.set_brightness(target)

    def restore(self) -> None:
        """Restore brightness saved by save_and_dim."""
        if self._saved_brightness is None:
            return
        self.set_brightness(self._saved_brightness)
        self._saved_brightness = None
