import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import QApplication
from chintu_ui import UIBridge
from chintu_assistant import ChintuAssistant

def test_assistant():
    print("=" * 60)
    print("  VERIFYING CHINTU SIRI-FOR-PC ASSISTANT LOGIC")
    print("=" * 60)

    app = QApplication(sys.argv)
    bridge = UIBridge()
    assistant = ChintuAssistant(bridge)

    # 1. Test PC Actions
    print("\n[TEST 1: PC System Commands]")
    print("Query: 'what time is it'")
    time_resp = assistant.handle_pc_action("what time is it")
    print(f"Response: {time_resp}")
    assert time_resp is not None, "Time command failed"

    print("\nQuery: 'what is today date'")
    date_resp = assistant.handle_pc_action("what is today date")
    print(f"Response: {date_resp}")
    assert date_resp is not None, "Date command failed"

    # 2. Test Dynamic Web Search & Summarization (No pre-setup Q&A)
    print("\n[TEST 2: Dynamic Live Web Search (Zero Hardcoded Answers)]")
    test_queries = [
        "who is the prime minister of india",
        "what is quantum computing",
        "who is the president of france",
        "capital of canada"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        resp = assistant.answer_web_query(q)
        print(f"Summary Answer: {resp}")
        assert resp and len(resp) > 5, f"Web query failed for '{q}'"

    # 3. Test Wake & Exit Phrases Detection
    print("\n[TEST 3: Wake and Exit Phrase Detection]")
    assert assistant.is_wake_word("hey chintu"), "Wake word 'hey chintu' failed"
    assert assistant.is_wake_word("chintu please"), "Wake word 'chintu please' failed"
    assert assistant.is_exit_phrase("exit"), "Exit phrase 'exit' failed"
    assert assistant.is_exit_phrase("go to sleep"), "Exit phrase 'go to sleep' failed"
    assert assistant.is_exit_phrase("goodbye chintu"), "Exit phrase 'goodbye chintu' failed"
    print("Wake words and Exit phrases detected successfully!")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED SUCCESSFULLY! CHINTU IS READY!")
    print("=" * 60)

if __name__ == "__main__":
    test_assistant()
