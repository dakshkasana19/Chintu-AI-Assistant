from chintu_assistant import ChintuAssistant
from chintu_ui import UIBridge
import sys
from PyQt6.QtWidgets import QApplication

def verify_knowledge():
    app = QApplication(sys.argv)
    bridge = UIBridge()
    assistant = ChintuAssistant(bridge)
    
    q1 = "who is the president of usa"
    q2 = "who is the president of india"
    
    print(f"Query 1: {q1}")
    print(f"Response 1: {assistant.get_smart_response(q1)}")
    
    print(f"\nQuery 2: {q2}")
    print(f"Response 2: {assistant.get_smart_response(q2)}")

if __name__ == "__main__":
    verify_knowledge()
