import sys
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QRadialGradient, QLinearGradient, QFont, QPainterPath

class UIBridge(QObject):
    status_changed = pyqtSignal(str)
    query_changed = pyqtSignal(str)
    response_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(bool)


class ChintuCoreUI(QMainWindow):
    """
    Siri for PC - Iridescent Glowing Voice Assistant Interface
    Features:
      - Frameless translucent floating overlay
      - Iridescent multi-layer pulsating Siri orb
      - Dynamic audio frequency wave visualizer
      - Live query transcription & spoken response HUD
      - Status indicator pill (Listening, Searching, Speaking, Sleeping)
    """
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

        # Window configuration: frameless, always-on-top, translucent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        
        # Dimensions
        self.width_ = 540
        self.height_ = 580
        self.resize(self.width_, self.height_)

        # Position at bottom-center of primary screen with nice margin
        self.reposition_window()

        # State & Animation variables
        self.angle = 0
        self.wave_offset = 0
        self.pulse = 0.0
        self.amp_mult = 1.0
        self.status_text = "STANDBY"
        self.query_text = ""
        self.response_text = ""
        
        # Connect bridge signals
        self.bridge.visibility_changed.connect(self.toggle_visibility)
        self.bridge.status_changed.connect(self.update_status)
        self.bridge.query_changed.connect(self.update_query)
        self.bridge.response_changed.connect(self.update_response)

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
            # Position at center-bottom of screen (Siri / Copilot style)
            x = screen_geo.x() + int((screen_geo.width() - self.width_) / 2)
            y = screen_geo.y() + screen_geo.height() - self.height_ - 40
            self.move(x, y)

    def toggle_visibility(self, visible):
        if visible:
            self.reposition_window()
            self.show()
            self.raise_()
            self.activateWindow()
            self.pulse = 0.0
            self.update()
        else:
            self.hide()

    def update_status(self, text):
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

    def update_query(self, text):
        self.query_text = text
        self.update()

    def update_response(self, text):
        self.response_text = text
        self.update()

    def update_animation(self):
        if not self.isVisible():
            return
        self.angle = (self.angle + 3) % 360
        self.wave_offset = (self.wave_offset + 6) % 1000
        self.pulse += 0.08
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = 190.0  # Center of orb

        # ------------------------------------------------------------------
        # 1. Background Glass Card Container
        # ------------------------------------------------------------------
        card_path = QPainterPath()
        card_rect = QRectF(15, 20, w - 30, h - 40)
        card_path.addRoundedRect(card_rect, 28, 28)

        # Translucent glass fill with subtle dark cyan gradient
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(10, 15, 30, 215))
        bg_grad.setColorAt(1.0, QColor(5, 8, 18, 240))
        painter.fillPath(card_path, bg_grad)

        # Subtle glowing glass border
        border_pen = QPen(QColor(0, 200, 255, 60), 1.5)
        painter.strokePath(card_path, border_pen)

        # ------------------------------------------------------------------
        # 2. Glowing Iridescent Siri Core Orb
        # ------------------------------------------------------------------
        pulse_val = 0.5 + 0.5 * np.sin(self.pulse)
        orb_radius = 110 + pulse_val * 14 * (self.amp_mult / 2.0)

        # Deep ambient aura
        aura_grad = QRadialGradient(cx, cy, orb_radius * 1.8)
        aura_grad.setColorAt(0.0, QColor(0, 180, 255, int(70 + 40 * pulse_val)))
        aura_grad.setColorAt(0.4, QColor(138, 43, 226, int(45 + 30 * pulse_val)))
        aura_grad.setColorAt(0.8, QColor(255, 20, 147, int(20 + 20 * pulse_val)))
        aura_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(aura_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(cx - orb_radius * 1.8, cy - orb_radius * 1.8, orb_radius * 3.6, orb_radius * 3.6))

        # Iridescent Core
        core_grad = QRadialGradient(cx - 15, cy - 15, orb_radius)
        core_grad.setColorAt(0.0, QColor(220, 250, 255, 230))
        core_grad.setColorAt(0.25, QColor(0, 220, 255, 200))
        core_grad.setColorAt(0.6, QColor(120, 50, 255, 170))
        core_grad.setColorAt(0.85, QColor(255, 0, 128, 140))
        core_grad.setColorAt(1.0, QColor(10, 20, 60, 0))
        painter.setBrush(core_grad)
        painter.drawEllipse(QRectF(cx - orb_radius, cy - orb_radius, orb_radius * 2, orb_radius * 2))

        # Multi-ring orbital rotation (Siri / AI resonance waves)
        ring_colors = [
            (QColor(0, 240, 255, 160), 1.0, 0),
            (QColor(240, 90, 255, 140), -1.2, 45),
            (QColor(80, 150, 255, 120), 0.8, 90),
        ]
        for color, speed, offset in ring_colors:
            ring_pen = QPen(color, 2.2)
            ring_pen.setDashPattern([6, 8])
            painter.setPen(ring_pen)
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(self.angle * speed + offset)
            r_scale = orb_radius * 0.95
            painter.drawEllipse(QRectF(-r_scale, -r_scale * 0.65, r_scale * 2, r_scale * 1.3))
            painter.restore()

        # ------------------------------------------------------------------
        # 3. Audio Frequency Waveform Visualizer
        # ------------------------------------------------------------------
        wave_y = cy + orb_radius + 20
        wave_width = 280
        start_x = cx - (wave_width / 2.0)
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(0, wave_width, 8):
            normalized_pos = (i - (wave_width / 2.0)) / (wave_width / 2.0)
            window = np.cos(normalized_pos * np.pi / 2.0) ** 2  # Hanning window for smooth edges
            
            wave_h = np.sin((i + self.wave_offset) * 0.14) * np.cos((i - self.wave_offset) * 0.08)
            bar_height = max(4.0, abs(wave_h) * 22.0 * self.amp_mult * window)
            
            # Bar color interpolation: Cyan -> Magenta -> Blue
            t = (i / float(wave_width))
            r = int(0 + t * 220)
            g = int(220 - t * 100)
            b = 255
            bar_color = QColor(r, g, b, int(160 + 80 * window))
            
            painter.setBrush(bar_color)
            painter.drawRoundedRect(
                QRectF(start_x + i, wave_y - (bar_height / 2.0), 4.5, bar_height),
                2.0, 2.0
            )

        # ------------------------------------------------------------------
        # 4. Status Badge Pill
        # ------------------------------------------------------------------
        badge_w = 170
        badge_h = 28
        badge_rect = QRectF(cx - (badge_w / 2.0), wave_y + 26, badge_w, badge_h)
        
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 14, 14)
        
        badge_bg = QColor(0, 180, 255, 30)
        badge_border = QColor(0, 210, 255, 120)
        if "LISTENING" in self.status_text:
            badge_bg = QColor(0, 230, 180, 45)
            badge_border = QColor(0, 255, 200, 170)
        elif "SPEAKING" in self.status_text:
            badge_bg = QColor(180, 60, 255, 45)
            badge_border = QColor(210, 100, 255, 170)
        elif "SEARCHING" in self.status_text:
            badge_bg = QColor(255, 170, 0, 45)
            badge_border = QColor(255, 190, 40, 170)

        painter.fillPath(badge_path, badge_bg)
        painter.setPen(QPen(badge_border, 1.2))
        painter.strokePath(badge_path, painter.pen())

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255, 230))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"● {self.status_text}")

        # ------------------------------------------------------------------
        # 5. Live Conversation HUD (Query & Response)
        # ------------------------------------------------------------------
        hud_top = wave_y + 64
        hud_rect = QRectF(35, hud_top, w - 70, h - hud_top - 48)

        # If user spoke something
        if self.query_text:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(160, 220, 255, 230))
            q_display = f'"{self.query_text}"'
            if len(q_display) > 85:
                q_display = q_display[:82] + "..."
            painter.drawText(
                QRectF(hud_rect.x(), hud_rect.y(), hud_rect.width(), 24),
                Qt.AlignmentFlag.AlignCenter,
                q_display
            )

        # Chintu response display
        if self.response_text:
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
            painter.setPen(QColor(235, 240, 255, 215))
            r_rect = QRectF(hud_rect.x(), hud_rect.y() + 28, hud_rect.width(), hud_rect.height() - 28)
            painter.drawText(
                r_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                self.response_text[:180] + ("..." if len(self.response_text) > 180 else "")
            )
        elif not self.query_text:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
            painter.setPen(QColor(160, 180, 210, 160))
            painter.drawText(
                hud_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Speak naturally. Ask any question or command."
            )

        # ------------------------------------------------------------------
        # 6. Bottom Sleep Mode Hint
        # ------------------------------------------------------------------
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor(130, 155, 190, 160))
        hint_rect = QRectF(20, h - 38, w - 40, 20)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter, 'Tip: Say "exit" or "sleep" to end conversation')
