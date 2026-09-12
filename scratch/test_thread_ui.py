import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import threading
from PyQt6.QtWidgets import QApplication
from chintu_ui import ChintuCoreUI, UIBridge

def test_thread():
    app = QApplication(sys.argv)
    b = UIBridge()
    w = ChintuCoreUI(b)

    def trigger():
        time.sleep(0.5)
        print("Emitting True from thread...")
        b.visibility_changed.emit(True)
        b.status_changed.emit("Listening...")
        time.sleep(0.5)
        print("Window isVisible in main loop:", w.isVisible())
        b.visibility_changed.emit(False)
        time.sleep(0.5)
        print("Window isVisible after false:", w.isVisible())
        app.quit()

    threading.Thread(target=trigger, daemon=True).start()
    app.exec()

if __name__ == "__main__":
    test_thread()
