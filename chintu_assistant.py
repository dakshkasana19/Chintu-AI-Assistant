import logging
import signal
import sys
import threading

from PyQt6.QtWidgets import QApplication

from chintu.audio import AudioCapture, AudioCaptureError, VoiceActivityDetector
from chintu.config import AssistantConfig, load_config
from chintu.logging_utils import configure_logging
from chintu.state import AssistantState, StateMachine
from chintu.ui.chintu_ui import ChintuCoreUI, UIBridge

logger = logging.getLogger("chintu")


class ChintuRuntime:
    """Run persistent microphone capture and VAD alongside the Qt UI."""

    def __init__(self, bridge, assistant_config: AssistantConfig | None = None):
        self.bridge = bridge
        self.config = assistant_config or load_config()
        self.state = StateMachine(AssistantState.SLEEPING)
        self._stopping = threading.Event()
        self._capture: AudioCapture | None = None
        self._audio_thread: threading.Thread | None = None
        logger.info("Chintu runtime initialized")

    def start(self) -> None:
        logger.info("Starting Chintu runtime")
        self.bridge.visibility_changed.emit(True)
        self.state.transition(AssistantState.SLEEPING)
        self._audio_thread = threading.Thread(
            target=self._audio_loop,
            name="chintu-audio",
            daemon=True,
        )
        self._audio_thread.start()

    def stop(self) -> None:
        logger.info("Stopping Chintu runtime")
        self._stopping.set()
        if self._capture is not None:
            self._capture.stop()
        if self._audio_thread is not None and self._audio_thread is not threading.current_thread():
            self._audio_thread.join(timeout=2.0)
            if self._audio_thread.is_alive():
                logger.warning("Audio worker did not stop within the shutdown timeout")
        self.bridge.visibility_changed.emit(False)
        self.state.transition(AssistantState.SLEEPING)

    def _audio_loop(self) -> None:
        try:
            detector = VoiceActivityDetector(self.config)
            logger.info("Voice activity detector ready: %s", detector.backend_name)
        except Exception as exc:
            logger.exception("Could not initialize a voice activity detector")
            self.bridge.status_changed.emit("Audio unavailable")
            self.bridge.response_changed.emit(
                f"Voice detection is unavailable: {exc}"
            )
            return

        self._capture = AudioCapture(
            sample_rate_hz=self.config.sample_rate_hz,
            frame_size_ms=self.config.frame_size_ms,
        )
        while not self._stopping.is_set():
            try:
                self._capture.start()
                detector.reset()
                self.bridge.status_changed.emit("Listening...")
                self.bridge.response_changed.emit(
                    "Microphone ready. Voice commands will be connected in later phases."
                )
                logger.info("Audio capture started; processing microphone frames")
                while not self._stopping.is_set():
                    frame = self._capture.read_frame(timeout=0.5)
                    if frame is None:
                        continue
                    self.bridge.microphone_rms_changed.emit(self._capture.rms)
                    segment = detector.process_frame(frame)
                    if segment is not None:
                        logger.info(
                            "VAD utterance captured: duration=%.2fs peak_rms=%.4f backend=%s",
                            segment.duration_seconds,
                            segment.peak_rms,
                            detector.backend_name,
                        )
                        self.bridge.status_changed.emit("Speech detected")
                        self.bridge.status_changed.emit("Listening...")
            except AudioCaptureError as exc:
                if self._stopping.is_set():
                    break
                logger.warning("Microphone stream interrupted; will retry: %s", exc)
                self.bridge.status_changed.emit("Microphone reconnecting...")
                self.bridge.response_changed.emit(
                    "Microphone unavailable. Check its connection; Chintu will retry."
                )
            except Exception as exc:
                logger.exception("Audio processing stopped unexpectedly")
                self.bridge.status_changed.emit("Audio error")
                self.bridge.response_changed.emit(
                    f"Audio processing stopped: {type(exc).__name__}."
                )
                return
            finally:
                self._capture.stop()

            self._stopping.wait(self.config.audio_recovery_seconds)
        logger.info("Audio worker stopped")


def handle_exit(*_args):
    logger.info("Exiting Chintu Assistant")
    QApplication.quit()
    sys.exit(0)


def main() -> None:
    config = load_config()
    configure_logging(config)
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)

    bridge = UIBridge()
    window = ChintuCoreUI(bridge)
    runtime = ChintuRuntime(bridge, config)
    runtime.start()
    app.aboutToQuit.connect(runtime.stop)

    logger.info("Loaded configuration: %s", config)
    logger.info("Chintu Phase 1 audio capture and VAD runtime started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
