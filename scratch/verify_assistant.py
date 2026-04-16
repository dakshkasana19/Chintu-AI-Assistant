from chintu_assistant import ChintuAssistant
from chintu_ui import UIBridge
import sys
from PyQt6.QtWidgets import QApplication

def verify_assistant():
    print("--- CHINTU ASSISTANT VERIFICATION ---")
    app = QApplication(sys.argv)
    bridge = UIBridge()
    assistant = ChintuAssistant(bridge)
    
    test_queries = [
        "who is the president of usa",
        "what is chatgpt",
        "when was genshin impact launched"
    ]
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        response = assistant.get_smart_response(query)
        print(f"Chintu Response: {response}")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_assistant()
