"""Unified Audio Service Orchestrator.

Combines Voice Activity Detection (VAD), Whisper Speech-to-Text, and
cloud observability into a cohesive, decoupled service layer.
"""

import time
from typing import Optional, Dict, Any

from src.audio.models import TranscribeResult, VADMetrics, AudioValidationResult
from src.audio.transcriber import WhisperTranscriber
from src.audio.vad import AudioVADEngine
from src.observability.tracer import tracer
from src.common.logging import term_log, debug_log, Colors


class AudioService:
    """Unified service orchestrating audio validation, VAD, STT, and telemetry."""

    def __init__(
        self,
        transcriber: Optional[WhisperTranscriber] = None,
        vad_engine: Optional[AudioVADEngine] = None
    ):
        """Initializes the AudioService.

        Args:
            transcriber: WhisperTranscriber instance (defaults to standard Groq client).
            vad_engine: AudioVADEngine instance (defaults to standard acoustic VAD).
        """
        self.transcriber = transcriber or WhisperTranscriber()
        self.vad = vad_engine or AudioVADEngine()

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "speech.wav",
        session_id: str = "voice-session",
        client_vad_metrics: Optional[VADMetrics] = None,
        filter_silence: bool = True
    ) -> TranscribeResult:
        """Processes audio through validation, VAD analysis, transcription, and telemetry.

        Args:
            audio_bytes: Raw binary content of the audio file.
            filename: Original filename (e.g. 'speech.wav').
            session_id: Correlation ID for Langfuse session tracing.
            client_vad_metrics: Optional VAD metrics computed on the edge (browser).
            filter_silence: If True, skips upstream Whisper inference when no speech is detected.

        Returns:
            TranscribeResult containing transcript, duration, and VAD metrics.

        Raises:
            ValueError: If audio validation fails (empty or corrupt).
        """
        t0 = time.time()
        term_log("🎙️ [AUDIO SERVICE]", f"Received '{filename}' ({len(audio_bytes)} bytes)", Colors.BLUE)

        # ---------------------------------------------------------------------
        # 1. Payload & Container Validation
        # ---------------------------------------------------------------------
        val: AudioValidationResult = self.vad.validate_payload(audio_bytes)
        if not val.valid:
            term_log("⚠️ [AUDIO REJECTED]", f"{val.error_message}", Colors.YELLOW)
            raise ValueError(val.error_message or "Invalid audio payload.")

        # ---------------------------------------------------------------------
        # 2. Voice Activity Detection (VAD)
        # ---------------------------------------------------------------------
        # Use client-provided VAD metrics if available; otherwise perform server-side VAD
        vad_metrics = client_vad_metrics or self.vad.analyze_audio(audio_bytes)

        # ---------------------------------------------------------------------
        # 3. Silence / Non-Speech Gate
        # ---------------------------------------------------------------------
        if filter_silence and not vad_metrics.is_speech:
            term_log(
                "🔇 [VAD GATE]",
                f"No speech detected (confidence={vad_metrics.confidence}). Skipping Whisper STT.",
                Colors.YELLOW
            )
            total_duration_s = round(time.time() - t0, 3)
            model_name = str(getattr(self.transcriber, "model", "whisper-large-v3-turbo"))
            result = TranscribeResult(
                text="",
                duration_s=total_duration_s,
                model=model_name,
                vad_metrics=vad_metrics,
                metadata={"skipped_reason": "No speech detected by VAD"}
            )
            self._log_telemetry(session_id, result, audio_bytes_len=len(audio_bytes))
            return result

        # ---------------------------------------------------------------------
        # 4. Speech-to-Text Transcription (In-Memory)
        # ---------------------------------------------------------------------
        result = self.transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            vad_metrics=vad_metrics
        )

        # ---------------------------------------------------------------------
        # 5. Cloud Observability Sync (Langfuse & Local JSON Traces)
        # ---------------------------------------------------------------------
        self._log_telemetry(session_id, result, audio_bytes_len=len(audio_bytes))

        return result

    def _log_telemetry(self, session_id: str, result: TranscribeResult, audio_bytes_len: int) -> None:
        """Syncs audio turn telemetry to Langfuse and local trace files."""
        try:
            metadata: Dict[str, Any] = {
                "stt_model": result.model,
                "stt_latency_s": result.duration_s,
                "payload_bytes": audio_bytes_len,
            }
            if result.vad_metrics:
                metadata.update({
                    "vad_is_speech": result.vad_metrics.is_speech,
                    "vad_confidence": result.vad_metrics.confidence,
                    "speech_duration_s": result.vad_metrics.speech_duration_s,
                    "silence_cutoff_s": result.vad_metrics.silence_duration_s,
                    "vad_latency_ms": result.vad_metrics.vad_latency_ms,
                })

            tracer.log_event(
                name="VoiceTurn",
                session_id=session_id,
                metadata=metadata,
                input_data=f"[Audio Payload: {audio_bytes_len} bytes]",
                output_data=result.text or "[Silence]"
            )
        except Exception as e:
            debug_log("⚠️ [DEBUG:TRACER_WARN]", f"Failed to log voice turn event: {e}")


# Global singleton instance for clean imports across the project
audio_service = AudioService()
