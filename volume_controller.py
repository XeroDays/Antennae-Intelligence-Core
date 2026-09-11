"""Windows system volume control with smoothing."""

from __future__ import annotations

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeController:
    """Get and set Windows master volume with exponential smoothing."""

    def __init__(self, smoothing_alpha: float = 0.12) -> None:
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_level: float | None = None
        self._volume = self._init_volume_interface()

    def set_volume(self, level: float) -> None:
        """Set master volume from a normalized level between 0.0 and 1.0."""
        clamped = max(0.0, min(1.0, level))

        if self._smoothed_level is None:
            self._smoothed_level = clamped
        else:
            alpha = self._smoothing_alpha
            self._smoothed_level = (alpha * clamped) + ((1.0 - alpha) * self._smoothed_level)

        self._volume.SetMasterVolumeLevelScalar(self._smoothed_level, None)

    def get_volume(self) -> float:
        """Return current master volume as a normalized level."""
        return float(self._volume.GetMasterVolumeLevelScalar())

    def get_smoothed_level(self) -> float:
        """Return the last smoothed level sent to the audio device."""
        if self._smoothed_level is None:
            return self.get_volume()
        return self._smoothed_level

    def _init_volume_interface(self) -> IAudioEndpointVolume:
        device = AudioUtilities.GetSpeakers()
        return device.EndpointVolume
