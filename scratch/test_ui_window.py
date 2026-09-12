import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from PyQt6.QtWidgets import QApplication
from chintu_ui import ChintuCoreUI, UIBridge

def test_ui():
    app = QApplication(sys.argv)
    bridge = UIBridge()
    window = ChintuCoreUI(bridge)
    
    print("Showing window...")
    bridge.visibility_changed.emit(True)
    bridge.status_changed.emit("Listening...")
    bridge.query_changed.emit("What is the capital of France?")
    bridge.response_changed.emit("Paris is the capital of France.")
    
    print("Geometry:", window.geometry().x(), window.geometry().y(), window.geometry().width(), window.geometry().height())
    print("Is visible?", window.isVisible())
    
    # Grab screenshot of window to verify rendering
    QApplication.processEvents()
    pixmap = window.grab()
    pixmap.save("scratch/ui_preview.png")
    print("Saved preview to scratch/ui_preview.png, size:", pixmap.width(), "x", pixmap.height())

if __name__ == "__main__":
    test_ui()
