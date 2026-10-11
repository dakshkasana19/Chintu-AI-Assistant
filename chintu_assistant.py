import logging
import signal
import sys

from PyQt6.QtWidgets import QApplication

from chintu.config import load_config
from chintu.logging_utils import configure_logging
from chintu.state import AssistantState, StateMachine
from chintu_ui import ChintuCoreUI, UIBridge

logger = configure_logging()
config = load_config()


class ChintuRuntime:
    """Thin runtime bootstrap for the Phase 0 local-first foundation."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.state = StateMachine(AssistantState.SLEEPING)
        logger.info("Chintu runtime initialized in sleeping state")

    def start(self) -> None:
        logger.info("Starting Chintu runtime")
        self.bridge.visibility_changed.emit(True)
        self.bridge.status_changed.emit("Standby")
        self.state.transition(AssistantState.SLEEPING)

    def stop(self) -> None:
        logger.info("Stopping Chintu runtime")
        self.bridge.visibility_changed.emit(False)
        self.state.transition(AssistantState.SLEEPING)


def handle_exit(*_args):
    logger.info("Exiting Chintu Assistant")
    QApplication.quit()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)

    bridge = UIBridge()
    window = ChintuCoreUI(bridge)
    runtime = ChintuRuntime(bridge)
    runtime.start()

    logger.info("Loaded configuration: %s", config)
    logger.info("Chintu assistant bootstrap completed")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
