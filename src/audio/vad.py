"""Voice Activity Detection (VAD) & Audio Validation Engine.

Provides acoustic speech analysis, container format validation, and silence
endpointing detection to prevent transcribing background silence or empty audio.
"""

import io
import time
import wave
from typing import Optional, Tuple
import numpy as np

from src.audio.models import VADMetrics, AudioValidationResult
from src.common.logging import term_log, debug_log, Colors


class AudioVADEngine:
    """Voice Activity Detector and Audio Payload Validator.

    Analyzes audio payloads to:
        1. Validate container headers (WAV, WebM, OGG, MP3) and payload size.
        2. Detect speech onset, speech offset, and trailing silence duration.
        3. Reject non-speech audio (pure silence, ambient room background) before STT.
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

    # Minimum payload size to be considered valid audio (e.g., WAV header is 44 bytes)
    MIN_AUDIO_BYTES = 44
    MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB max payload (Whisper limit)

    def __init__(
        self,
        energy_threshold: float = 0.015,
        silence_cutoff_s: float = 0.6,
        frame_duration_ms: int = 30
    ):
        """Initializes the VAD engine.

        Args:
            energy_threshold: RMS volume cutoff separating vocal speech from ambient silence.
            silence_cutoff_s: Minimum trailing silence duration in seconds to trigger turn endpoint.
            frame_duration_ms: Analysis frame window size in milliseconds.
        """
        self.energy_threshold = energy_threshold
        self.silence_cutoff_s = silence_cutoff_s
        self.frame_duration_ms = frame_duration_ms

    def validate_payload(self, audio_bytes: bytes) -> AudioValidationResult:
        """Validates raw audio bytes before passing to downstream engines.

        Args:
            audio_bytes: Binary payload received from client or network.

        Returns:
            AudioValidationResult with validation status, size, and container format.
        """
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

        # Detect container format via header magic bytes
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

    def analyze_audio(self, audio_bytes: bytes) -> VADMetrics:
        """Analyzes audio payload for voice activity, speech ratio, and trailing silence.

        Args:
            audio_bytes: Complete audio binary payload (WAV preferred).

        Returns:
            VADMetrics with speech confirmation, duration, and latency.
        """
        t0 = time.time()

        # Extract PCM samples
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

        total_duration = len(samples) / float(sample_rate)
        frame_size = int(sample_rate * (self.frame_duration_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 480

        # Frame-level RMS energy analysis
        speech_frames = 0
        total_frames = 0
        trailing_silence_frames = 0
        active_speech_encountered = False

        for i in range(0, len(samples), frame_size):
            frame = samples[i : i + frame_size]
            if len(frame) == 0:
                continue

            total_frames += 1
            # Calculate Root Mean Square (RMS) energy
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

        # Calculate confidence as speech ratio with non-linear boost
        speech_ratio = speech_frames / max(total_frames, 1)
        is_speech = speech_frames > 2 and speech_ratio > 0.15
        confidence = round(min(1.0, speech_ratio * 1.5) if is_speech else 0.0, 2)

        debug_log(
            "🔍 [DEBUG:VAD]",
            f"VAD Analyzed: duration={total_duration:.2f}s, speech={speech_duration_s:.2f}s, "
            f"trailing_silence={trailing_silence_s:.2f}s, is_speech={is_speech} in {vad_latency}ms"
        )

        return VADMetrics(
            is_speech=is_speech,
            confidence=confidence,
            speech_duration_s=speech_duration_s,
            silence_duration_s=trailing_silence_s,
            vad_latency_ms=vad_latency
        )

    def _extract_pcm_samples(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Extracts normalized Float32 PCM samples from WAV payload or raw bytes.

        Returns:
            Tuple of (numpy array of float32 samples [-1.0, 1.0], sample_rate).
        """
        # Try standard WAV header parsing
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw_frames = wav_file.readframes(n_frames)

                if sample_width == 2:  # 16-bit PCM
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:  # 8-bit unsigned
                    samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sample_width == 4:  # 32-bit float or int
                    samples = np.frombuffer(raw_frames, dtype=np.float32)
                else:
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                # Downmix stereo/multichannel to mono
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)

                return samples, sample_rate
        except Exception:
            # Fallback for raw PCM or non-WAV formats
            try:
                # Strip potential 44-byte container header if present
                raw_data = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                return samples, 16000
            except Exception:
                return np.array([], dtype=np.float32), 16000
