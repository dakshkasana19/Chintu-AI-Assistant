import math
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen


class UIBridge(QObject):
    status_changed = pyqtSignal(str)
    query_changed = pyqtSignal(str)
    response_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(bool)
    microphone_rms_changed = pyqtSignal(float)


class ChintuCoreUI(QMainWindow):
    """
    Joyful, minimalist floating voice assistant indicator.
    Features:
      - Completely frameless, transparent floating ring (no box, no shadows, no borders).
      - Warm buttercup yellow (#FFE550) dashed circular boundary with rounded caps.
      - Positioned directly in the center of the screen.
      - Smooth clockwise rotation while active.
      - Expands outward dynamically when voice is detected, naturally widening dash gaps.
    """

    COLOR = QColor("#FFE550")
    NUM_DASHES = 16
    DASH_ARC_LEN = 22.0
    BASE_RADIUS = 142.5
    MAX_EXPANSION = 45.0

    def __init__(self, bridge: UIBridge):
        super().__init__()
        self.bridge = bridge

        # Window configuration: frameless, no drop shadow, always-on-top, fully transparent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent; border: none;")

        # Canvas sized for the larger central circle
        self.width_ = 440
        self.height_ = 440
        self.resize(self.width_, self.height_)

        # Position at dead center of screen
        self.reposition_window()

        # State & Animation variables
        self.angle = 0.0
        self.expansion = 0.0
        self.pulse = 0.0
        self.amp_mult = 1.0
        self.status_text = "STANDBY"
        self.query_text = ""
        self.response_text = ""
        self.microphone_rms = 0.0

        # Connect bridge signals
        self.bridge.visibility_changed.connect(self.toggle_visibility)
        self.bridge.status_changed.connect(self.update_status)
        self.bridge.query_changed.connect(self.update_query)
        self.bridge.response_changed.connect(self.update_response)
        self.bridge.microphone_rms_changed.connect(self.update_microphone_rms)

        # 60 FPS animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        # Start hidden in sleep mode
        self.hide()

    def reposition_window(self):
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Exact dead center of primary screen
            x = screen_geo.x() + int((screen_geo.width() - self.width_) / 2)
            y = screen_geo.y() + int((screen_geo.height() - self.height_) / 2)
            self.move(x, y)

    def toggle_visibility(self, visible: bool):
        if visible:
            self.reposition_window()
            self.show()
            self.raise_()
            self.expansion = 0.0
            self.update()
        else:
            self.hide()

    def update_status(self, text: str):
        self.status_text = text.upper()
        if "LISTENING" in self.status_text:
            self.amp_mult = 3.6
        elif "SPEAKING" in self.status_text:
            self.amp_mult = 3.0
        elif "SEARCHING" in self.status_text or "THINKING" in self.status_text or "PROCESSING" in self.status_text:
            self.amp_mult = 2.0
        else:
            self.amp_mult = 1.2
        self.update()

    def update_query(self, text: str):
        self.query_text = text
        self.update()

    def update_response(self, text: str):
        self.response_text = text
        self.update()

    def update_microphone_rms(self, value: float):
        self.microphone_rms = max(0.0, float(value))
        self.update()

    def update_animation(self):
        if not self.isVisible():
            return
        # Clockwise rotation: positive angle increment in standard Qt coordinates
        self.angle = (self.angle + 1.2) % 360.0

        # Voice expansion smoothing
        target_boost = min(1.0, self.microphone_rms * 18.0)
        self.expansion += (target_boost - self.expansion) * 0.25

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0

        # Dynamic radius based on microphone volume
        radius = self.BASE_RADIUS + self.expansion * self.MAX_EXPANSION

        # Fixed arc length in degrees decreases as radius expands, widening the gaps!
        span_deg = (self.DASH_ARC_LEN / max(1.0, radius)) * (180.0 / math.pi)

        # Flat buttercup yellow pen with friendly rounded caps (doubled thickness)
        pen = QPen(self.COLOR, 11.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rect = QRectF(-radius, -radius, radius * 2.0, radius * 2.0)
        span_16 = int(round(span_deg * 16))
        step_deg = 360.0 / self.NUM_DASHES

        painter.save()
        painter.translate(cx, cy)
        # Clockwise rotation
        painter.rotate(self.angle)

        for _ in range(self.NUM_DASHES):
            # Qt drawArc uses 1/16ths of a degree; center dash around local 0
            painter.drawArc(rect, int(round(-span_deg / 2.0 * 16)), span_16)
            painter.rotate(step_deg)

        painter.restore()
