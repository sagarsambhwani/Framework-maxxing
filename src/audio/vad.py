"""Voice Activity Detection (VAD) & Audio Validation Engine.

Combines high-accuracy Machine Learning (Silero VAD ONNX neural network)
with mathematical energy analysis (RMS fallback) for low-latency,
noise-resilient turn-taking endpointing.
"""

import io
import os
import time
import wave
from typing import Optional, Tuple, Dict, Any
import numpy as np

from src.audio.models import VADMetrics, AudioValidationResult
from src.common.logging import term_log, debug_log, Colors

# Optional ONNX Runtime import for ML VAD
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


class AudioVADEngine:
    """Voice Activity Detector and Audio Payload Validator.

    Supports dual-engine detection:
        1. Primary: Silero VAD v5 ONNX Neural Network (~1ms latency, noise-immune).
        2. Fallback: Fast Root Mean Square (RMS) Acoustic Energy calculation.
    """

    # Magic byte signatures for audio container formats
    AUDIO_SIGNATURES = {
        b"RIFF": "wav",
        b"\x1a\x45\xdf\xa3": "webm",
        b"OggS": "ogg",
        b"ID3": "mp3",
        b"\xff\xfb": "mp3",
        b"\xff\xf3": "mp3",
        b"\xff\xf2": "mp3",
    }

    MIN_AUDIO_BYTES = 44
    MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB max payload

    DEFAULT_WEIGHTS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights", "silero_vad.onnx"
    )

    def __init__(
        self,
        energy_threshold: float = 0.015,
        ml_threshold: float = 0.5,
        silence_cutoff_s: float = 0.65,
        frame_duration_ms: int = 30,
        mode: str = "auto",
        model_path: Optional[str] = None
    ):
        """Initializes the VAD engine.

        Args:
            energy_threshold: RMS cutoff for mathematical fallback.
            ml_threshold: Probability threshold (0.0 to 1.0) for ML voice detection.
            silence_cutoff_s: Trailing silence duration triggering turn endpoint.
            frame_duration_ms: Frame analysis window size.
            mode: Detection mode: 'auto' (ML with energy fallback), 'ml', or 'energy'.
            model_path: Optional custom path to silero_vad.onnx weights.
        """
        self.energy_threshold = energy_threshold
        self.ml_threshold = ml_threshold
        self.silence_cutoff_s = silence_cutoff_s
        self.frame_duration_ms = frame_duration_ms
        self.mode = mode.lower()
        self.model_path = model_path or self.DEFAULT_WEIGHTS_PATH

        self._session = None
        self._init_ml_session()

        # Streaming state for real-time live microphone frames
        self.reset_streaming_state()

    def _init_ml_session(self) -> None:
        """Loads the Silero ONNX neural network session if available."""
        if self.mode == "energy" or not ONNX_AVAILABLE:
            return

        if os.path.exists(self.model_path):
            try:
                self._session = ort.InferenceSession(self.model_path)
                debug_log("🔍 [DEBUG:VAD_INIT]", f"Loaded Silero VAD ML model from {self.model_path}")
            except Exception as e:
                debug_log("⚠️ [DEBUG:VAD_WARN]", f"Failed to initialize Silero ONNX: {e}. Falling back to Energy VAD.")
                self._session = None

    @property
    def is_ml_active(self) -> bool:
        """Returns True if the ML neural network is loaded and operational."""
        return self._session is not None and self.mode in ("auto", "ml")

    def reset_streaming_state(self) -> None:
        """Resets the recurrent neural state for a fresh streaming turn."""
        self._stream_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._stream_context = np.zeros((1, 64), dtype=np.float32)

    def validate_payload(self, audio_bytes: bytes) -> AudioValidationResult:
        """Validates raw audio bytes before passing to downstream engines."""
        if not audio_bytes or len(audio_bytes) == 0:
            return AudioValidationResult(
                valid=False,
                size_bytes=0,
                format="empty",
                error_message="Audio payload is empty (0 bytes)."
            )

        size = len(audio_bytes)
        if size < self.MIN_AUDIO_BYTES:
            return AudioValidationResult(
                valid=False,
                size_bytes=size,
                format="corrupt",
                error_message=f"Payload too small ({size} bytes). Minimum {self.MIN_AUDIO_BYTES} bytes required."
            )

        if size > self.MAX_AUDIO_BYTES:
            return AudioValidationResult(
                valid=False,
                size_bytes=size,
                format="oversized",
                error_message=f"Payload exceeds 25MB maximum limit ({size / (1024 * 1024):.1f} MB)."
            )

        detected_format = "unknown"
        for sig, fmt in self.AUDIO_SIGNATURES.items():
            if audio_bytes.startswith(sig):
                detected_format = fmt
                break

        return AudioValidationResult(
            valid=True,
            size_bytes=size,
            format=detected_format,
            error_message=None
        )

    def step_frame(self, pcm_chunk_512: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """Runs single-frame inference on a 32ms (512 samples @ 16kHz) audio slice.

        Args:
            pcm_chunk_512: Float32 numpy array with 512 audio samples.
            sample_rate: Audio sampling rate (default 16000).

        Returns:
            Tuple of (is_speech: bool, confidence: float 0.0-1.0).
        """
        # 1. Primary: Silero ML Neural Network
        if self.is_ml_active:
            try:
                frame_data = pcm_chunk_512.reshape(1, -1).astype(np.float32)
                x = np.concatenate([self._stream_context, frame_data], axis=1) # 576 samples
                self._stream_context = x[:, -64:]
                sr = np.array(sample_rate, dtype=np.int64)

                out, self._stream_state = self._session.run(
                    None,
                    {"input": x, "state": self._stream_state, "sr": sr}
                )
                prob = float(out[0][0])
                is_speech = prob >= self.ml_threshold
                return is_speech, prob
            except Exception:
                pass

        # 2. Fallback: RMS Energy calculation
        rms = float(np.sqrt(np.mean(pcm_chunk_512 ** 2))) if len(pcm_chunk_512) > 0 else 0.0
        is_speech = rms >= self.energy_threshold
        conf = min(1.0, rms * 10.0) if is_speech else 0.0
        return is_speech, conf

    def analyze_audio(self, audio_bytes: bytes) -> VADMetrics:
        """Analyzes full audio payload for voice activity, speech ratio, and trailing silence."""
        t0 = time.time()
        samples, sample_rate = self._extract_pcm_samples(audio_bytes)

        if len(samples) == 0:
            vad_latency = round((time.time() - t0) * 1000, 2)
            return VADMetrics(
                is_speech=False,
                confidence=0.0,
                speech_duration_s=0.0,
                silence_duration_s=0.0,
                vad_latency_ms=vad_latency
            )

        # ---------------------------------------------------------------------
        # Path A: Silero ML Neural Network Evaluation
        # ---------------------------------------------------------------------
        if self.is_ml_active:
            try:
                metrics = self._analyze_with_silero(samples, sample_rate, t0)
                # In auto mode, if ML rejected (e.g. synthetic test tone), verify energy
                if self.mode == "auto" and not metrics.is_speech:
                    energy_metrics = self._analyze_with_energy(samples, sample_rate, t0)
                    if energy_metrics.is_speech and energy_metrics.confidence >= 0.5:
                        return energy_metrics
                return metrics
            except Exception as e:
                debug_log("⚠️ [DEBUG:VAD_FALLBACK]", f"ML inference failed: {e}. Using energy fallback.")

        # ---------------------------------------------------------------------
        # Path B: Acoustic Energy (RMS) Evaluation
        # ---------------------------------------------------------------------
        return self._analyze_with_energy(samples, sample_rate, t0)

    def _analyze_with_silero(self, samples: np.ndarray, sample_rate: int, t0: float) -> VADMetrics:
        """Evaluates audio with Silero ONNX neural network across 512-sample frames."""
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        sr = np.array(16000, dtype=np.int64)

        chunk_size = 512
        speech_frames = 0
        total_frames = 0
        trailing_silence_frames = 0
        active_speech_encountered = False
        probs = []

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad final chunk to 512 samples
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")

            total_frames += 1
            x = np.concatenate([context, chunk.reshape(1, -1).astype(np.float32)], axis=1)
            context = x[:, -64:]

            out, state = self._session.run(None, {"input": x, "state": state, "sr": sr})
            prob = float(out[0][0])
            probs.append(prob)

            if prob >= self.ml_threshold:
                speech_frames += 1
                trailing_silence_frames = 0
                active_speech_encountered = True
            else:
                if active_speech_encountered:
                    trailing_silence_frames += 1

        vad_latency = round((time.time() - t0) * 1000, 2)
        frame_duration_s = chunk_size / 16000.0
        speech_duration_s = round(speech_frames * frame_duration_s, 3)
        trailing_silence_s = round(trailing_silence_frames * frame_duration_s, 3)

        peak_prob = max(probs) if probs else 0.0
        is_speech = speech_frames >= 2 and peak_prob >= self.ml_threshold
        confidence = round(peak_prob, 2) if is_speech else 0.0

        debug_log(
            "🔍 [DEBUG:VAD_ML]",
            f"Silero VAD: speech={speech_duration_s:.2f}s, silence={trailing_silence_s:.2f}s, "
            f"peak_prob={peak_prob:.2f}, is_speech={is_speech} in {vad_latency}ms"
        )

        return VADMetrics(
            is_speech=is_speech,
            confidence=confidence,
            speech_duration_s=speech_duration_s,
            silence_duration_s=trailing_silence_s,
            vad_latency_ms=vad_latency
        )

    def _analyze_with_energy(self, samples: np.ndarray, sample_rate: int, t0: float) -> VADMetrics:
        """Evaluates audio with mathematical Root Mean Square (RMS) energy."""
        frame_size = int(sample_rate * (self.frame_duration_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 480

        speech_frames = 0
        total_frames = 0
        trailing_silence_frames = 0
        active_speech_encountered = False

        for i in range(0, len(samples), frame_size):
            frame = samples[i : i + frame_size]
            if len(frame) == 0:
                continue

            total_frames += 1
            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms >= self.energy_threshold:
                speech_frames += 1
                trailing_silence_frames = 0
                active_speech_encountered = True
            else:
                if active_speech_encountered:
                    trailing_silence_frames += 1

        vad_latency = round((time.time() - t0) * 1000, 2)
        frame_duration_s = self.frame_duration_ms / 1000.0
        speech_duration_s = round(speech_frames * frame_duration_s, 3)
        trailing_silence_s = round(trailing_silence_frames * frame_duration_s, 3)

        speech_ratio = speech_frames / max(total_frames, 1)
        is_speech = speech_frames > 2 and speech_ratio > 0.15
        confidence = round(min(1.0, speech_ratio * 1.5) if is_speech else 0.0, 2)

        return VADMetrics(
            is_speech=is_speech,
            confidence=confidence,
            speech_duration_s=speech_duration_s,
            silence_duration_s=trailing_silence_s,
            vad_latency_ms=vad_latency
        )

    def _extract_pcm_samples(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Extracts normalized Float32 PCM samples from WAV payload or raw bytes."""
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw_frames = wav_file.readframes(n_frames)

                if sample_width == 2:
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:
                    samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sample_width == 4:
                    samples = np.frombuffer(raw_frames, dtype=np.float32)
                else:
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)

                return samples, sample_rate
        except Exception:
            try:
                raw_data = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                return samples, 16000
            except Exception:
                return np.array([], dtype=np.float32), 16000
