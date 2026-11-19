import logging
import os
import queue
import signal
import sys
import threading
import time

from PyQt6.QtWidgets import QApplication

from chintu.actions import ActionResult, Intent, IntentRouter, WindowsActionExecutor, WindowsAppIndex
from chintu.audio import (
    AudioCapture,
    AudioCaptureError,
    SpeechSegment,
    VoiceActivityDetector,
    VoskWhisperWakeWordDetector,
    WakeWordDetector,
)
from chintu.asr import FasterWhisperASR, TranscriptionResult
from chintu.config import AssistantConfig, load_config
from chintu.logging_utils import configure_logging
from chintu.rag import RAGPipeline
from chintu.single_instance import acquire_single_instance
from chintu.state import AssistantState, StateMachine
from chintu.tts import TextToSpeech
from chintu.ui.chintu_ui import ChintuCoreUI, UIBridge

logger = logging.getLogger("chintu")


class ChintuRuntime:
    """Run persistent capture, VAD, local wake detection, command ASR, actions, TTS, and RAG."""

    def __init__(self, bridge, assistant_config: AssistantConfig | None = None):
        self.bridge = bridge
        self.config = assistant_config or load_config()
        self.state = StateMachine(AssistantState.SLEEPING)
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._wake_stopping = threading.Event()
        self._command_stopping = threading.Event()
        self._wake_segments: queue.Queue[SpeechSegment] = queue.Queue(maxsize=4)
        self._command_segments: queue.Queue[SpeechSegment] = queue.Queue(maxsize=4)
        self._capture: AudioCapture | None = None
        self._audio_thread: threading.Thread | None = None
        self._wake_thread: threading.Thread | None = None
        self._command_thread: threading.Thread | None = None
        self._asr: FasterWhisperASR | None = None
        self._tts: TextToSpeech | None = None
        self._rag: RAGPipeline | None = None
        self._tts_barge_in = threading.Event()
        self._intent_router = IntentRouter(self.config.exit_phrases)
        self._actions: WindowsActionExecutor | None = None
        self._pending_confirmation: tuple[Intent, float] | None = None
        self.latest_transcription: TranscriptionResult | None = None
        self._tts_echo_gate_until: float = 0.0  # monotonic timestamp; discard segments before this
        logger.info("Chintu runtime initialized")

    def start(self) -> None:
        logger.info("Starting Chintu runtime")
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
        self._command_stopping.set()
        if self._capture is not None:
            self._capture.stop()
        if self._command_thread is not None and self._command_thread is not threading.current_thread():
            self._command_thread.join(timeout=5.0)
            if self._command_thread.is_alive():
                logger.warning("Command ASR worker did not stop within the shutdown timeout")
        if self._wake_thread is not None and self._wake_thread is not threading.current_thread():
            self._wake_thread.join(timeout=5.0)
            if self._wake_thread.is_alive():
                logger.warning("Wake-word worker did not stop within the shutdown timeout")
        if self._audio_thread is not None and self._audio_thread is not threading.current_thread():
            self._audio_thread.join(timeout=2.0)
            if self._audio_thread.is_alive():
                logger.warning("Audio worker did not stop within the shutdown timeout")
        self.bridge.visibility_changed.emit(False)
        if self._tts is not None:
            self._tts.stop(timeout=2.0)
        if self._actions is not None:
            canceled = self._actions.cancel_timers()
            if canceled:
                logger.info("Canceled %d active timers during shutdown", canceled)
        self._transition_state(AssistantState.SLEEPING)

    def _transition_state(self, next_state: AssistantState) -> None:
        with self._state_lock:
            if self.state.state is not next_state:
                self.state.transition(next_state)

    def _on_speaking_start(self) -> None:
        with self._state_lock:
            if self.state.state in {AssistantState.PROCESSING, AssistantState.LISTENING}:
                self.state.transition(AssistantState.SPEAKING)
        self.bridge.status_changed.emit("Speaking")

    def _on_speaking_end(self) -> None:
        with self._state_lock:
            if self.state.state is AssistantState.SPEAKING:
                self.state.transition(AssistantState.LISTENING)
        if self._tts_barge_in.is_set():
            self._tts_barge_in.clear()
            self._tts_echo_gate_until = 0.0
        else:
            self._tts_echo_gate_until = time.monotonic() + 0.8
        self.bridge.status_changed.emit("Listening for command")

    def _interrupt_tts_for_speech(self) -> None:
        with self._state_lock:
            if self.state.state is not AssistantState.SPEAKING:
                return
            self.state.transition(AssistantState.LISTENING)

        self._tts_barge_in.set()
        if self._tts is not None:
            self._tts.interrupt()
        self._tts_echo_gate_until = 0.0
        self.bridge.status_changed.emit("Listening for command")

    def _respond(self, message: str, *, speak: bool = True) -> None:
        self.bridge.response_changed.emit(message)
        if speak and self._tts is not None and self.config.tts_enabled:
            self._tts.speak(message)

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

        try:
            shared_model = (
                wake_detector.whisper_model
                if self.config.asr_model_size == "base"
                else None
            )
            self._asr = FasterWhisperASR(self.config, model=shared_model)
        except Exception as exc:
            self._asr = None
            logger.exception("Could not initialize command ASR")
            self.bridge.status_changed.emit("Command transcription unavailable")
            self.bridge.response_changed.emit(
                f"Command transcription is unavailable: {exc}"
            )

        self._actions = WindowsActionExecutor(
            WindowsAppIndex.discover(self.config.app_aliases),
            notify=self._notify_timer,
        )

        if self.config.tts_enabled and self._tts is None:
            try:
                self._tts = TextToSpeech(
                    config=self.config,
                    on_speaking_start=self._on_speaking_start,
                    on_speaking_end=self._on_speaking_end,
                )
            except Exception as exc:
                self._tts = None
                logger.warning("Could not initialize TTS: %s", exc)

        if self.config.rag_enabled and self._rag is None:
            try:
                self._rag = RAGPipeline(self.config)
            except Exception as exc:
                self._rag = None
                logger.warning("Could not initialize RAG pipeline: %s", exc)
        if self._stopping.is_set():
            return
        self._wake_stopping.clear()
        self._command_stopping.clear()
        self._wake_thread = threading.Thread(
            target=self._wake_loop,
            args=(wake_detector,),
            name="chintu-wake",
            daemon=True,
        )
        self._wake_thread.start()
        if self._asr is not None:
            self._command_thread = threading.Thread(
                target=self._command_asr_loop,
                args=(self._asr,),
                name="chintu-command-asr",
                daemon=True,
            )
            self._command_thread.start()
        self._capture = AudioCapture(
            sample_rate_hz=self.config.sample_rate_hz,
            frame_size_ms=self.config.frame_size_ms,
        )
        try:
            while not self._stopping.is_set():
                try:
                    self._capture.start()
                    detector.reset()
                    self.bridge.status_changed.emit("Standby - say Chintu")
                    self.bridge.response_changed.emit(
                        'Waiting for the wake phrase "Chintu".'
                    )
                    logger.info("Audio capture started; processing microphone frames")
                    while not self._stopping.is_set():
                        frame = self._capture.read_frame(timeout=0.5)
                        if frame is None:
                            continue
                        self.bridge.microphone_rms_changed.emit(self._capture.rms)
                        segment = detector.process_frame(frame)
                        if detector.speech_active:
                            self._interrupt_tts_for_speech()
                        if segment is not None:
                            logger.info(
                                "VAD utterance captured: duration=%.2fs peak_rms=%.4f backend=%s",
                                segment.duration_seconds,
                                segment.peak_rms,
                                detector.backend_name,
                            )
                            self._queue_speech_segment(segment)
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
            self._command_stopping.set()
            if self._wake_thread is not None:
                self._wake_thread.join(timeout=5.0)
                if self._wake_thread.is_alive():
                    logger.warning("Wake-word worker did not stop within the shutdown timeout")
            if self._command_thread is not None:
                self._command_thread.join(timeout=5.0)
                if self._command_thread.is_alive():
                    logger.warning("Command ASR worker did not stop within the shutdown timeout")
        logger.info("Audio worker stopped")

    def _queue_speech_segment(self, segment: SpeechSegment) -> None:
        self._interrupt_tts_for_speech()
        with self._state_lock:
            current_state = self.state.state
            if current_state is AssistantState.SPEAKING:
                self._tts_barge_in.set()
                if self._tts is not None:
                    self._tts.interrupt()
                self._tts_echo_gate_until = 0.0
                self.state.transition(AssistantState.LISTENING)
                if self._asr is None:
                    return
                target_queue = self._command_segments
                purpose = "Command ASR (barge-in)"
            elif current_state is AssistantState.SLEEPING:
                target_queue = self._wake_segments
                purpose = "Wake confirmation"
            elif current_state in {
                AssistantState.LISTENING,
                AssistantState.PROCESSING,
            }:
                if self._asr is None:
                    return
                # Suppress echo: discard segments captured right after TTS finished
                if time.monotonic() < self._tts_echo_gate_until:
                    logger.debug("Discarding segment within TTS echo gate window")
                    return
                target_queue = self._command_segments
                purpose = "Command ASR"
            else:
                return
            try:
                target_queue.put_nowait(segment)
            except queue.Full:
                logger.warning("%s is busy; discarded a VAD speech segment", purpose)
                self.bridge.status_changed.emit(f"{purpose} busy")

    def _queue_wake_confirmation(self, segment: SpeechSegment) -> None:
        self._queue_speech_segment(segment)

    def _wake_loop(self, detector: WakeWordDetector) -> None:
        while not self._wake_stopping.is_set() and not self._stopping.is_set():
            try:
                segment = self._wake_segments.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._state_lock:
                if not self.state.is_sleeping():
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
            # Flush any segments queued before the wake transition — these
            # contain the wake-word utterance itself and must not be treated
            # as commands.
            while not self._command_segments.empty():
                try:
                    self._command_segments.get_nowait()
                except Exception:
                    break
            self.bridge.visibility_changed.emit(True)
            if self._asr is None:
                self.bridge.status_changed.emit("Command transcription unavailable")
                self.bridge.response_changed.emit(
                    "Wake word confirmed, but local command transcription is unavailable."
                )
            else:
                self.bridge.status_changed.emit("Listening - wake word confirmed")
                self.bridge.response_changed.emit(
                    "Wake word confirmed. Listening for your command."
                )
                self._respond("Yes?", speak=True)

    def _command_asr_loop(self, asr: FasterWhisperASR) -> None:
        while not self._command_stopping.is_set() and not self._stopping.is_set():
            try:
                segment = self._command_segments.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._state_lock:
                if self.state.state not in {
                    AssistantState.LISTENING,
                    AssistantState.PROCESSING,
                }:
                    continue
                if self.state.state is AssistantState.LISTENING:
                    self.state.transition(AssistantState.PROCESSING)
            self.bridge.status_changed.emit("Processing command")
            self.latest_transcription = None
            try:
                result = asr.transcribe(segment)
            except Exception:
                logger.exception("Command transcription failed")
                self.bridge.status_changed.emit("Command transcription error")
                self.bridge.response_changed.emit(
                    "I couldn't transcribe that command. Please try again."
                )
            else:
                self.latest_transcription = result
                logger.info(
                    "Command transcription %s: reason=%s avg_logprob=%s "
                    "no_speech_prob=%s compression_ratio=%s",
                    "accepted" if result.accepted else "rejected",
                    result.reason,
                    result.avg_logprob,
                    result.no_speech_prob,
                    result.compression_ratio,
                )
                if result.accepted:
                    self.bridge.query_changed.emit(result.raw_text)
                    try:
                        self._handle_command(result.normalized_text)
                    except Exception:
                        logger.exception("Command routing or action failed")
                        self.bridge.response_changed.emit(
                            "I couldn't complete that local action. Please try again."
                        )
                else:
                    self.bridge.response_changed.emit(
                        f"Didn't catch a clear command ({result.reason}). Please try again."
                    )
            finally:
                current_state = self.state.state
                with self._state_lock:
                    if (
                        self.state.state is AssistantState.PROCESSING
                        and not self._stopping.is_set()
                    ):
                        self.state.transition(AssistantState.LISTENING)
                if not self._stopping.is_set():
                    if current_state is AssistantState.SLEEPING:
                        self.bridge.status_changed.emit("Standby - say Chintu")
                    else:
                        self.bridge.status_changed.emit("Listening for command")

    def _handle_command(self, command_text: str) -> None:
        intent = self._intent_router.route(command_text)
        logger.info(
            "Intent routed: name=%s confidence=%.2f slot_names=%s",
            intent.name,
            intent.confidence,
            sorted(intent.slots),
        )
        pending = self._pending_confirmation
        if pending is not None and time.monotonic() > pending[1]:
            self._pending_confirmation = None
            pending = None

        if intent.name == "confirm":
            if pending is None:
                self._respond("There is no action waiting for confirmation.")
                return
            self._pending_confirmation = None
            self._execute_action(pending[0], confirmed=True)
            return
        if intent.name == "cancel":
            self._pending_confirmation = None
            self._respond("Canceled.")
            return
        if pending is not None:
            self._pending_confirmation = None
            self._respond(
                "The pending action was canceled because a different command was received."
            )

        if intent.name == "sleep":
            with self._state_lock:
                if self.state.state is not AssistantState.SLEEPING:
                    self.state.transition(AssistantState.SLEEPING)
            self.bridge.visibility_changed.emit(False)
            self.bridge.status_changed.emit("Standby - say Chintu")
            self._respond("Going to standby.", speak=False)
            if self._tts is not None:
                self._tts.interrupt()
            return
        if intent.name == "information_request":
            query = str(intent.slots.get("query") or command_text)
            self._answer_information_request(query)
            return
        if intent.name in {"unknown", "clarify"}:
            prompt = intent.slots.get("prompt")
            self._respond(
                str(prompt) if prompt else "I didn't understand that command. Please try again."
            )
            return
        if self._actions is None:
            self._respond("Local actions are unavailable.")
            return
        if intent.requires_confirmation:
            self._pending_confirmation = (intent, time.monotonic() + 30.0)
            target = intent.slots.get("target") or intent.slots.get("operation", "action")
            self._respond(
                f"This may close or power off your computer ({target}). "
                "Say 'confirm' to proceed or 'cancel' to stop."
            )
            return
        self._execute_action(intent)

    def _answer_information_request(self, query: str) -> None:
        if self._rag is None:
            self._respond("Information retrieval is not connected.")
            return

        with self._state_lock:
            if self.state.state is not AssistantState.PROCESSING:
                self.state.transition(AssistantState.PROCESSING)

        self.bridge.status_changed.emit("Searching...")
        try:
            rag_result = self._rag.answer(query)
            logger.info(
                "RAG completed: verified=%s confidence=%.2f answer_chars=%d",
                rag_result.verified,
                rag_result.confidence,
                len(rag_result.answer),
            )
            response = rag_result.answer
            if rag_result.verified and rag_result.source_title:
                response = f"{response} Source: {rag_result.source_title}."
            self._respond(response)
        except Exception:
            logger.exception("RAG information retrieval failed")
            self._respond("I couldn't look up that information. Please try again.")

    def _execute_action(self, intent: Intent, *, confirmed: bool = False) -> None:
        if self._actions is None:
            self._respond("Local actions are unavailable.")
            return
        result = self._actions.execute(intent, confirmed=confirmed)
        self._publish_action_result(intent, result)

    def _publish_action_result(self, intent: Intent, result: ActionResult) -> None:
        if result.confirmation_required:
            self._pending_confirmation = (intent, time.monotonic() + 30.0)
        if intent.name == "system_power" and result.success:
            self.bridge.status_changed.emit("Power action scheduled")
        elif intent.name == "lock_workstation" and result.success:
            self.bridge.status_changed.emit("Locking workstation")
        self._respond(result.message)
        logger.info(
            "Action %s: success=%s message=%s",
            intent.name,
            result.success,
            result.message,
        )

    def _notify_timer(self, message: str) -> None:
        logger.info("Scheduled notification: %s", message)
        self.bridge.status_changed.emit("Timer finished")
        self._respond(message)


def handle_exit(*_args):
    logger.info("Exiting Chintu Assistant")
    QApplication.quit()
    sys.exit(0)


def main() -> None:
    instance_guard = acquire_single_instance()
    if instance_guard is None:
        print("Chintu is already running; this launch was ignored.", file=sys.stderr)
        return

    try:
        config = load_config()
        configure_logging(config)
        signal.signal(signal.SIGINT, handle_exit)
        app = QApplication(sys.argv)

        bridge = UIBridge()
        window = ChintuCoreUI(bridge)
        runtime = ChintuRuntime(bridge, config)
        runtime.start()
        app.aboutToQuit.connect(runtime.stop)

        logger.info("Starting Chintu process (pid=%s)", os.getpid())
        logger.info("Loaded configuration: %s", config)
        logger.info("Chintu Phase 3 audio, wake-word, and command-ASR runtime started")
        sys.exit(app.exec())
    finally:
        instance_guard.close()


if __name__ == "__main__":
    main()
