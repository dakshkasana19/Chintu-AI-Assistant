import logging
import queue
import signal
import sys
import threading

from PyQt6.QtWidgets import QApplication

from chintu.audio import (
    AudioCapture,
    AudioCaptureError,
    SpeechSegment,
    VoiceActivityDetector,
    VoskWhisperWakeWordDetector,
    WakeWordDetector,
)
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
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._wake_stopping = threading.Event()
        self._wake_segments: queue.Queue[SpeechSegment] = queue.Queue(maxsize=1)
        self._capture: AudioCapture | None = None
        self._audio_thread: threading.Thread | None = None
        self._wake_thread: threading.Thread | None = None
        logger.info("Chintu runtime initialized")

    def start(self) -> None:
        logger.info("Starting Chintu runtime")
        self.bridge.visibility_changed.emit(True)
        self._transition_state(AssistantState.SLEEPING)
        self._audio_thread = threading.Thread(
            target=self._audio_loop,
            name="chintu-audio",
            daemon=True,
        )
        self._audio_thread.start()

    def stop(self) -> None:
        logger.info("Stopping Chintu runtime")
        self._stopping.set()
        self._wake_stopping.set()
        if self._capture is not None:
            self._capture.stop()
        if self._wake_thread is not None and self._wake_thread is not threading.current_thread():
            self._wake_thread.join(timeout=5.0)
            if self._wake_thread.is_alive():
                logger.warning("Wake-word worker did not stop within the shutdown timeout")
        if self._audio_thread is not None and self._audio_thread is not threading.current_thread():
            self._audio_thread.join(timeout=2.0)
            if self._audio_thread.is_alive():
                logger.warning("Audio worker did not stop within the shutdown timeout")
        self.bridge.visibility_changed.emit(False)
        self._transition_state(AssistantState.SLEEPING)

    def _transition_state(self, next_state: AssistantState) -> None:
        with self._state_lock:
            if self.state.state is not next_state:
                self.state.transition(next_state)

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

        try:
            wake_detector = VoskWhisperWakeWordDetector(self.config)
        except Exception as exc:
            logger.exception("Could not initialize the wake-word detector")
            self.bridge.status_changed.emit("Wake detection unavailable")
            self.bridge.response_changed.emit(
                f"Wake-word detection is unavailable: {exc}"
            )
            return

        if self._stopping.is_set():
            return
        self._wake_stopping.clear()
        self._wake_thread = threading.Thread(
            target=self._wake_loop,
            args=(wake_detector,),
            name="chintu-wake",
            daemon=True,
        )
        self._wake_thread.start()
        self._capture = AudioCapture(
            sample_rate_hz=self.config.sample_rate_hz,
            frame_size_ms=self.config.frame_size_ms,
        )
        try:
            while not self._stopping.is_set():
                try:
                    self._capture.start()
                    detector.reset()
                    self.bridge.status_changed.emit("Listening for Chintu...")
                    self.bridge.response_changed.emit(
                        "Listening locally for the configured wake phrase."
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
                            self._queue_wake_confirmation(segment)
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
        finally:
            self._wake_stopping.set()
            if self._wake_thread is not None:
                self._wake_thread.join(timeout=5.0)
                if self._wake_thread.is_alive():
                    logger.warning("Wake-word worker did not stop within the shutdown timeout")
        logger.info("Audio worker stopped")

    def _queue_wake_confirmation(self, segment: SpeechSegment) -> None:
        with self._state_lock:
            sleeping = self.state.is_sleeping()
        if not sleeping:
            return
        try:
            self._wake_segments.put_nowait(segment)
        except queue.Full:
            logger.warning("Wake confirmation is busy; discarded a VAD speech segment")

    def _wake_loop(self, detector: WakeWordDetector) -> None:
        while not self._wake_stopping.is_set() and not self._stopping.is_set():
            try:
                segment = self._wake_segments.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                match = detector.detect(segment)
            except Exception:
                logger.exception("Wake-word confirmation failed")
                self.bridge.status_changed.emit("Wake confirmation error")
                self.bridge.response_changed.emit(
                    "Wake-word confirmation failed; Chintu remains asleep."
                )
                continue
            if match is None or self._stopping.is_set():
                continue
            with self._state_lock:
                if self.state.is_sleeping():
                    self.state.transition(AssistantState.WAKE_CONFIRM)
                    self.state.transition(AssistantState.LISTENING)
                else:
                    continue
            self.bridge.status_changed.emit("Wake word confirmed")
            self.bridge.response_changed.emit(
                "Wake word confirmed. Command processing will be connected in a later phase."
            )


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
    logger.info("Chintu Phase 2 audio capture, VAD, and wake-word runtime started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
