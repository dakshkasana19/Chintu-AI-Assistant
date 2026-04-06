import sys
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QRadialGradient, QFont

class UIBridge(QObject):
    status_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(bool)

class ChintuCoreUI(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.resize(700, 700)
        self.angle = 0
        self.status_text = "CHINTU: STANDBY"
        self.wave_offset = 0
        self.amp_mult = 1.0
        self.pulse = 0
        self.hide()
        
        self.bridge.visibility_changed.connect(self.toggle_visibility)
        self.bridge.status_changed.connect(self.update_status)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(20)
        
    def toggle_visibility(self, visible):
        if visible:
            self.show()
            self.raise_()
            self.activateWindow()
            self.pulse = 0
        else:
            self.hide()

    def update_animation(self):
        if not self.isVisible(): return
        self.angle += 4
        self.wave_offset += 8
        self.pulse += 0.12
        self.update()

    def update_status(self, text):
        self.status_text = f"CHINTU: {text.upper()}"
        if "LISTENING" in self.status_text: 
            self.amp_mult = 4.0
        elif "SPEAKING" in self.status_text: 
            self.amp_mult = 2.8
        elif "SEARCHING" in self.status_text: 
            self.amp_mult = 2.2
        elif "PROCESSING" in self.status_text:
            self.amp_mult = 1.8
        else: 
            self.amp_mult = 1.4

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        cx, cy = center.x(), center.y()
        
        # Dynamic pulsing core - FIXED QColor
        pulse_alpha = int(50 + 40 * abs(np.sin(self.pulse)))
        pulse_alpha_mid = int(pulse_alpha / 1.5)  # ✅ FIXED: int division
        
        grad = QRadialGradient(cx, cy, 320)
        grad.setColorAt(0, QColor(0, 160, 255, pulse_alpha))
        grad.setColorAt(0.5, QColor(0, 120, 220, pulse_alpha_mid))  # ✅ int alpha
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - 320, cy - 320, 640, 640)
        
        # Multi-ring orbital animation
        for i, alpha in enumerate([200, 160, 120, 80]):
            pen = QPen(QColor(80, 240 + i*10, 255, alpha), 3.5)
            pen.setDashPattern([8, 12])
            painter.setPen(pen)
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(self.angle * 1.1 + i * 45)
            painter.drawEllipse(-240 - i*25, -240 - i*25, 480 + i*50, 480 + i*50)
            painter.restore()
        
        # Advanced VAD audio visualization
        pen = QPen(QColor(0, 280, 255, 240), 3.5)
        painter.setPen(pen)
        for i in range(0, 420, 12):
            wave = np.sin((i + self.wave_offset) * 0.16) * (28 * self.amp_mult)
            painter.drawLine(cx - 210 + i, cy + 200, cx - 210 + i, cy + 200 - int(abs(wave)))
        
        # Double-glow status text
        painter.setPen(QColor(255, 255, 255, 140))
        painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        painter.drawText(self.rect().adjusted(0, 0, 0, -50), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, self.status_text)
        
        painter.setPen(QColor(100, 260, 255, 255))
        painter.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        painter.drawText(self.rect().adjusted(0, 0, 0, -20), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, self.status_text)
