"""Terminal Turn-Taking Conversational Voice Agent.

Provides hands-free, full-duplex conversational voice AI directly in the terminal:
    1. Real-time microphone capture via SoundDevice (16kHz 16-bit mono).
    2. Silero VAD ML neural network running on 32ms frames.
    3. Conversational Turn State Machine (LISTENING, USER_SPEAKING, PROCESSING, AI_SPEAKING).
    4. Instant Barge-In (interruption of AI speech when user begins talking).
    5. Sub-200ms Speech-to-Text via Groq Whisper Large v3 Turbo.
    6. Low-latency streaming token completions via Groq LPU / LiteLLM.
    7. Asynchronous audio speech synthesis via Windows SAPI voice.
"""

import io
import sys
import time
import wave
import queue
import threading
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import win32com.client
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

from src.audio.service import audio_service
from src.audio.vad import AudioVADEngine
from src.gateway.router import gateway
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors


class TurnState(Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AI_SPEAKING = "AI_SPEAKING"


class TerminalVoiceAgent:
    """Conversational Turn-Taking Voice Agent running directly in the terminal."""

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512  # 32ms frames at 16kHz

    def __init__(
        self,
        model: str = "groq/qwen/qwen3.8-27b",
        silence_timeout_ms: int = 650,
        enable_tts: bool = True,
        barge_in: bool = False
    ):
        self.model = model
        self.silence_timeout_ms = silence_timeout_ms
        self.silence_frames_threshold = int(silence_timeout_ms / 32)
        self.enable_tts = enable_tts and WIN32_AVAILABLE
        self.barge_in = barge_in

        self.vad = AudioVADEngine()
        self.state = TurnState.LISTENING

        self.audio_frames: List[np.ndarray] = []
        self.silence_count = 0
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a concise, helpful, intelligent AI voice assistant. Keep answers natural, engaging, and brief (2-3 sentences) for spoken dialogue."}
        ]

        self.audio_queue = queue.Queue()
        self.running = False
        self.barge_in_event = threading.Event()

        # Initialize Windows SAPI TTS voice
        self.speaker = None
        if self.enable_tts:
            try:
                self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
                self.speaker.Rate = 1  # Conversational cadence
            except Exception as e:
                debug_log("⚠️ [DEBUG:TTS_WARN]", f"Failed to initialize SAPI voice: {e}")
                self.enable_tts = False

    def stop_ai_audio(self) -> None:
        """Immediately interrupts and silences active AI speech output."""
        if self.enable_tts and self.speaker:
            try:
                # Flag 2 = SVSFPurgeBeforeSpeak: stops currently speaking voice instantly
                self.speaker.Speak("", 2)
            except Exception:
                pass
        self.barge_in_event.set()

    def speak_and_wait(self, text: str) -> None:
        """Speaks text and monitors for completion or barge-in."""
        if not self.enable_tts or not self.speaker:
            return
        clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
        if not clean_text:
            return

        try:
            # Start speech asynchronously (Flag 1 = SVSFlagsAsync)
            self.speaker.Speak(clean_text, 1)
            time.sleep(0.08)

            while self.speaker.Status.RunningState == 2 and self.running:
                if self.barge_in:
                    try:
                        frame = self.audio_queue.get(timeout=0.03)
                        is_speech, conf = self.vad.step_frame(frame)
                        if is_speech and conf > 0.7:
                            term_log("🛑 [BARGE-IN]", f"Interruption detected ({conf*100:.1f}%)! Silencing...", Colors.YELLOW)
                            self.stop_ai_audio()
                            break
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.05)
        except Exception as e:
            debug_log("⚠️ [DEBUG:TTS_ERROR]", f"SAPI playback error: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio hardware streaming callback called every 32ms."""
        frame = indata[:, 0].copy()
        self.audio_queue.put(frame)

    def run(self) -> None:
        """Main event loop managing the turn-taking state machine."""
        if sd is None:
            print(f"{Colors.RED}❌ Error: 'sounddevice' package is required. Install with: uv pip install sounddevice{Colors.END}")
            return

        barge_mode_str = f"{Colors.GREEN}ENABLED (Headphones Mode){Colors.END}" if self.barge_in else f"{Colors.YELLOW}Safe Turn-Taking (Open Speakers Safe){Colors.END}"

        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}{Colors.GREEN}🎙️  CHATGPT PRO TERMINAL VOICE AGENT (Hands-Free Duplex){Colors.END}")
        print(f"👉 Model         : {Colors.YELLOW}{self.model}{Colors.END}")
        print(f"👉 VAD Engine    : {'Silero ONNX Neural Network (ML Active)' if self.vad.is_ml_active else 'Acoustic RMS Energy'}")
        print(f"👉 Turn Mode     : {barge_mode_str}")
        print(f"👉 Voice Output  : {'Active (Windows SAPI Voice)' if self.enable_tts else 'Disabled (Text-only)'}")
        print(f"👉 Silence Pause : {self.silence_timeout_ms}ms")
        print(f"👉 Exit          : Press {Colors.BOLD}Ctrl+C{Colors.END} to quit")
        print("=" * 80 + "\n")

        self.running = True
        term_log("🎙️  [LISTENING]", "Waiting for you to speak...", Colors.BLUE)

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                blocksize=self.CHUNK_SIZE,
                dtype="float32",
                callback=self._audio_callback
            ):
                while self.running:
                    try:
                        frame = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    self._process_frame(frame)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Voice session ended by user. Goodbye!{Colors.END}\n")
        finally:
            self.running = False
            self.stop_ai_audio()

    def _process_frame(self, frame: np.ndarray) -> None:
        """Processes a single 32ms audio frame through VAD and state transitions."""
        if self.state == TurnState.AI_SPEAKING:
            return

        is_speech, confidence = self.vad.step_frame(frame)

        # ---------------------------------------------------------------------
        # 1. State: LISTENING (Speech Onset Detection)
        # ---------------------------------------------------------------------
        if self.state == TurnState.LISTENING:
            if is_speech:
                term_log("🗣️  [SPEECH DETECTED]", f"Voice active ({confidence*100:.1f}%), recording turn...", Colors.CYAN)
                self.state = TurnState.USER_SPEAKING
                self.audio_frames = [frame]
                self.silence_count = 0

        # ---------------------------------------------------------------------
        # 2. State: USER_SPEAKING (Accumulation & Endpointing)
        # ---------------------------------------------------------------------
        elif self.state == TurnState.USER_SPEAKING:
            self.audio_frames.append(frame)

            if is_speech:
                self.silence_count = 0
            else:
                self.silence_count += 1

                # If user paused for longer than silence threshold -> Turn finished!
                if self.silence_count >= self.silence_frames_threshold:
                    self.state = TurnState.PROCESSING
                    self._handle_user_turn_completed()

    def _handle_user_turn_completed(self) -> None:
        """Assembles recorded speech, runs Whisper STT, and queries LLM."""
        term_log("⏳ [TURN ENDPOINT]", f"Silence detected ({self.silence_timeout_ms}ms). Processing...", Colors.GREEN)

        if not self.audio_frames or len(self.audio_frames) < 10:
            self.audio_frames = []
            self.silence_count = 0
            self.state = TurnState.LISTENING
            return

        # 1. Encode in-memory 16kHz WAV
        raw_samples = np.concatenate(self.audio_frames)
        wav_bytes = self._encode_wav(raw_samples)
        self.audio_frames = []
        self.silence_count = 0

        # 2. Fast Speech-to-Text via AudioService
        import asyncio
        try:
            result = asyncio.run(audio_service.transcribe(wav_bytes, filename="cli_turn.wav"))
            user_text = result.text.strip()
        except Exception as e:
            term_log("❌ [STT ERROR]", f"{e}", Colors.RED)
            user_text = ""

        if not user_text:
            term_log("🔇 [NO SPEECH]", "Audio did not contain recognizable speech.", Colors.GRAY)
            self.state = TurnState.LISTENING
            self.vad.reset_streaming_state()
            return

        print(f"\n{Colors.BOLD}{Colors.CYAN}👤 You:{Colors.END} {Colors.BOLD}{user_text}{Colors.END}\n")

        # 3. Stream AI Completion & Speak Response
        self.state = TurnState.AI_SPEAKING
        self.barge_in_event.clear()
        self.vad.reset_streaming_state()

        self.conversation_history.append({"role": "user", "content": user_text})
        print(f"{Colors.BOLD}{Colors.GREEN}🤖 Assistant ({self.model}):{Colors.END} ", end="", flush=True)

        full_reply = ""
        try:
            for chunk in gateway.stream(model=self.model, messages=self.conversation_history, max_tokens=250):
                if self.barge_in_event.is_set():
                    print(f" {Colors.YELLOW}[Interrupted]{Colors.END}", flush=True)
                    break
                print(chunk, end="", flush=True)
                full_reply += chunk
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")

        print("\n")

        # If not interrupted, speak full response and wait for playback
        if not self.barge_in_event.is_set() and full_reply:
            self.conversation_history.append({"role": "assistant", "content": full_reply})
            if self.enable_tts and self.speaker:
                self.speak_and_wait(full_reply)

        # Drain any residual audio frames collected during processing / speaker playback
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset VAD streaming state so previous speaker sounds don't linger in context buffer
        self.vad.reset_streaming_state()
        self.state = TurnState.LISTENING
        term_log("🎙️  [LISTENING]", "Waiting for your next question...", Colors.BLUE)

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        """Encodes float32 numpy audio into 16kHz 16-bit PCM WAV bytes in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            pcm16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()


def run_voice(
    model: str = "groq/qwen/qwen3.8-27b",
    silence_timeout_ms: int = 650,
    voice_output: bool = True,
    barge_in: bool = False
) -> None:
    """Entrypoint function for CLI voice command."""
    agent = TerminalVoiceAgent(
        model=model,
        silence_timeout_ms=silence_timeout_ms,
        enable_tts=voice_output,
        barge_in=barge_in
    )
    agent.run()
