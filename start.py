"""Antennae Core - hand gesture tracker entry point."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from gesture_detector import DEFAULT_PINCH_THRESHOLD
from hand_tracker import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_PRESENCE_CONFIDENCE,
    DEFAULT_TRACKING_CONFIDENCE,
    HandTracker,
)
from volume_controller import VolumeController


QUALITY_OPTIONS = {
    "480p (640x480)": (640, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
}

# Moving hands 50% of frame width apart = +100% volume swing.
VOLUME_SENSITIVITY = 2.0

CONFIDENCE_TOOLTIPS = {
    "detection": (
        "Min confidence to detect a new hand in a frame. "
        "Higher = fewer false hands detected."
    ),
    "presence": (
        "Min confidence to keep a detected hand visible between frames. "
        "Higher = ghost hands drop faster."
    ),
    "tracking": (
        "Min confidence to keep tracking an already-found hand. "
        "Lower = more forgiving tracking."
    ),
}


class Tooltip:
    """Show a delayed tooltip on hover."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip_window: tk.Toplevel | None = None
        self._after_id: str | None = None

        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._hide()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        if self._tip_window is not None:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip_window = tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            padx=6,
            pady=4,
            wraplength=280,
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None


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

        self.quality_var = tk.StringVar(value="480p (640x480)")
        quality_dropdown = ttk.Combobox(
            toolbar,
            textvariable=self.quality_var,
            values=list(QUALITY_OPTIONS.keys()),
            state="readonly",
            width=22,
        )
        quality_dropdown.pack(side=tk.LEFT, padx=(8, 0))
        quality_dropdown.bind("<<ComboboxSelected>>", self._on_quality_changed)

        ttk.Label(toolbar, text="Pinch Threshold:").pack(side=tk.LEFT, padx=(16, 0))

        self.pinch_threshold_var = tk.DoubleVar(value=DEFAULT_PINCH_THRESHOLD)
        pinch_scale = ttk.Scale(
            toolbar,
            from_=0.02,
            to=0.20,
            orient=tk.HORIZONTAL,
            variable=self.pinch_threshold_var,
            length=120,
            command=self._on_pinch_threshold_changed,
        )
        pinch_scale.pack(side=tk.LEFT, padx=(8, 0))

        self.pinch_label_var = tk.StringVar(value=f"{DEFAULT_PINCH_THRESHOLD:.2f}")
        ttk.Label(toolbar, textvariable=self.pinch_label_var, width=5).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self.status_var = tk.StringVar(value="Starting camera...")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        self._build_confidence_row()

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_confidence_row(self) -> None:
        """Build the hand detection confidence settings container."""
        container = ttk.LabelFrame(self, text="Hand Detection Confidence", padding=(10, 6))
        container.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.conf_detection_var = tk.DoubleVar(value=DEFAULT_DETECTION_CONFIDENCE)
        self.conf_presence_var = tk.DoubleVar(value=DEFAULT_PRESENCE_CONFIDENCE)
        self.conf_tracking_var = tk.DoubleVar(value=DEFAULT_TRACKING_CONFIDENCE)

        self.conf_detection_label_var = tk.StringVar(
            value=f"{DEFAULT_DETECTION_CONFIDENCE:.2f}"
        )
        self.conf_presence_label_var = tk.StringVar(
            value=f"{DEFAULT_PRESENCE_CONFIDENCE:.2f}"
        )
        self.conf_tracking_label_var = tk.StringVar(
            value=f"{DEFAULT_TRACKING_CONFIDENCE:.2f}"
        )

        self._add_confidence_control(
            container,
            "Detection",
            self.conf_detection_var,
            self.conf_detection_label_var,
            CONFIDENCE_TOOLTIPS["detection"],
        )
        self._add_confidence_control(
            container,
            "Presence",
            self.conf_presence_var,
            self.conf_presence_label_var,
            CONFIDENCE_TOOLTIPS["presence"],
        )
        self._add_confidence_control(
            container,
            "Tracking",
            self.conf_tracking_var,
            self.conf_tracking_label_var,
            CONFIDENCE_TOOLTIPS["tracking"],
        )

    def _add_confidence_control(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.DoubleVar,
        value_label_var: tk.StringVar,
        tooltip_text: str,
    ) -> None:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))

        info_icon = tk.Label(
            parent,
            text="ⓘ",
            cursor="question_arrow",
            foreground="#4aa3ff",
        )
        info_icon.pack(side=tk.LEFT, padx=(0, 6))
        Tooltip(info_icon, tooltip_text)

        ttk.Scale(
            parent,
            from_=0.50,
            to=0.99,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=100,
            command=self._on_confidence_changed,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(parent, textvariable=value_label_var, width=5).pack(
            side=tk.LEFT, padx=(0, 16)
        )

    def _on_confidence_changed(self, _value: str | None = None) -> None:
        detection = round(self.conf_detection_var.get(), 2)
        presence = round(self.conf_presence_var.get(), 2)
        tracking = round(self.conf_tracking_var.get(), 2)

        self.conf_detection_label_var.set(f"{detection:.2f}")
        self.conf_presence_label_var.set(f"{presence:.2f}")
        self.conf_tracking_label_var.set(f"{tracking:.2f}")

        self.tracker.reconfigure(detection, presence, tracking)

    def _on_quality_changed(self, _event=None) -> None:
        self._apply_camera_quality(self.quality_var.get())

    def _on_pinch_threshold_changed(self, _value: str | None = None) -> None:
        value = round(self.pinch_threshold_var.get(), 2)
        self.pinch_label_var.set(f"{value:.2f}")
        self.tracker.set_pinch_threshold(value)

    def _apply_camera_quality(self, selection: str) -> None:
        width, height = QUALITY_OPTIONS.get(selection, (640, 480))
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

            is_interlocked = self.tracker.detect_and_draw_interlocked(frame, hand_results)
            if is_interlocked:
                self.volume_ctrl.set_volume_instant(0.0)
                self._vol_baseline_dist = None
                self._vol_baseline_volume = None
                frame = self._draw_volume_bar(frame, 0.0)
                self.status_var.set("Interlocked | VOL 0%")
            else:
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

                    if hand_results:
                        labels = ", ".join(
                            f"{hand.handedness}: {hand.gesture}" for hand in hand_results
                        )
                        self.status_var.set(labels)
                    else:
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
