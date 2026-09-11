"""Antennae Core - hand gesture tracker entry point."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from hand_tracker import HandTracker
from volume_controller import VolumeController


QUALITY_OPTIONS = {
    "480p (640x480)": (640, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
}

# Moving hands 50% of frame width apart = +100% volume swing.
VOLUME_SENSITIVITY = 2.0


class HandGestureApp(tk.Tk):
    """Resizable window with camera quality selection and hand tracking."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Antennae Core - Hand Gesture Tracker")
        self.geometry("900x620")
        self.minsize(640, 480)
        self.configure(bg="#1e1e1e")

        self.tracker = HandTracker()
        self.volume_ctrl = VolumeController()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._photo_image: ImageTk.PhotoImage | None = None
        self._running = True
        self._vol_baseline_dist: float | None = None
        self._vol_baseline_volume: float | None = None

        self._build_ui()
        self._apply_camera_quality(self.quality_var.get())
        self._update_frame()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Camera Quality:").pack(side=tk.LEFT)

        self.quality_var = tk.StringVar(value="720p (1280x720)")
        quality_dropdown = ttk.Combobox(
            toolbar,
            textvariable=self.quality_var,
            values=list(QUALITY_OPTIONS.keys()),
            state="readonly",
            width=22,
        )
        quality_dropdown.pack(side=tk.LEFT, padx=(8, 0))
        quality_dropdown.bind("<<ComboboxSelected>>", self._on_quality_changed)

        self.status_var = tk.StringVar(value="Starting camera...")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_quality_changed(self, _event=None) -> None:
        self._apply_camera_quality(self.quality_var.get())

    def _apply_camera_quality(self, selection: str) -> None:
        width, height = QUALITY_OPTIONS.get(selection, (1280, 720))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status_var.set(f"Camera: {actual_width}x{actual_height}")

    def _on_canvas_resize(self, _event=None) -> None:
        # Canvas resize is handled during frame rendering.
        return

    def _update_frame(self) -> None:
        if not self._running:
            return

        ok, frame = self.cap.read()
        if ok:
            hand_results = self.tracker.process(frame)
            frame = self.tracker.draw(frame, hand_results)

            raw_dist = self.tracker.draw_volume_bridge(frame, hand_results)
            if raw_dist is not None:
                if self._vol_baseline_dist is None:
                    self._vol_baseline_dist = raw_dist
                    self._vol_baseline_volume = self.volume_ctrl.get_volume()

                delta = raw_dist - self._vol_baseline_dist
                new_volume = self._vol_baseline_volume + (delta * VOLUME_SENSITIVITY)
                self.volume_ctrl.set_volume(new_volume)

                frame = self._draw_volume_bar(frame, self.volume_ctrl.get_smoothed_level())
                volume_percent = int(self.volume_ctrl.get_smoothed_level() * 100)
                status = f"Volume control active | VOL {volume_percent}%"
                if hand_results:
                    labels = ", ".join(
                        f"{hand.handedness}: {hand.gesture}" for hand in hand_results
                    )
                    self.status_var.set(f"{status} | {labels}")
                else:
                    self.status_var.set(status)
            else:
                self._vol_baseline_dist = None
                self._vol_baseline_volume = None

            if raw_dist is None and hand_results:
                labels = ", ".join(
                    f"{hand.handedness}: {hand.gesture}" for hand in hand_results
                )
                self.status_var.set(labels)
            elif raw_dist is None:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.status_var.set(f"Camera: {width}x{height} | No hands detected")

            self._render_frame(frame)
        else:
            self.status_var.set("Failed to read from camera.")

        self.after(15, self._update_frame)

    def _draw_volume_bar(self, frame: np.ndarray, level: float) -> np.ndarray:
        """Draw a vertical volume bar on the right side of the frame."""
        output = frame.copy()
        frame_height, frame_width = output.shape[:2]

        bar_width = 24
        bar_height = int(frame_height * 0.45)
        margin = 20
        x1 = frame_width - margin - bar_width
        y1 = int((frame_height - bar_height) / 2)
        x2 = x1 + bar_width
        y2 = y1 + bar_height

        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 40, 40), -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)

        fill_height = int(bar_height * max(0.0, min(1.0, level)))
        if fill_height > 0:
            fill_y1 = y2 - fill_height
            cv2.rectangle(output, (x1 + 2, fill_y1), (x2 - 2, y2 - 2), (255, 255, 255), -1)

        label = f"VOL {int(level * 100)}%"
        cv2.putText(
            output,
            label,
            (x1 - 10, y2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _render_frame(self, frame: np.ndarray) -> None:
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)

        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        target_width = max(1, int(frame_width * scale))
        target_height = max(1, int(frame_height * scale))

        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self._photo_image = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        x = (canvas_width - target_width) // 2
        y = (canvas_height - target_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._photo_image)

    def _on_close(self) -> None:
        self._running = False
        self.cap.release()
        self.tracker.close()
        self.destroy()


def main() -> None:
    app = HandGestureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
