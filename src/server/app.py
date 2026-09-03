"""FastAPI Application Factory & Endpoints."""
import os
import time
import json
import tempfile
from typing import Generator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from src.server.models import ChatStreamRequest
from src.gateway.router import gateway
from src.guardrails.rails_manager import guardrails
from src.observability.tracer import tracer
from src.common.config import settings
from src.common.logging import term_log, Colors


def create_app() -> FastAPI:
    """Creates and configures the FastAPI web application."""
    app = FastAPI(title="ChatGPT Pro Multi-Model API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Static Frontend
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir = os.path.join(root_dir, "static")

    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h2>Frontend static files not found.</h2>")

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest):
        term_log("📥 [REQUEST]", f"Session: {req.session_id} | Model: {Colors.YELLOW}{req.model}{Colors.END}", Colors.BLUE)
        term_log("💬 [PROMPT]", f"'{req.prompt[:100]}...'", Colors.CYAN)

        # 1. NeMo Guardrails Input Check
        if req.guardrails_enabled:
            guard = guardrails.validate_input(req.prompt)
            if not guard["allowed"]:
                tracer.log_event("Guardrail:Blocked", req.session_id, {"reason": guard["reason"]}, input_data=req.prompt)
                async def blocked_stream():
                    yield f"data: {json.dumps({'blocked': True, 'reason': guard['reason']})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(blocked_stream(), media_type="text/event-stream")
            clean_prompt = guard["clean_prompt"]
        else:
            clean_prompt = req.prompt

        # 2. Build Messages
        messages = [{"role": "system", "content": "You are ChatGPT Pro, a helpful, intelligent, and concise AI assistant."}]
        for turn in req.history:
            if turn.get("content"):
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content")})
        messages.append({"role": "user", "content": clean_prompt})

        # 3. Stream Tokens via Gateway
        async def event_stream():
            start_t = time.time()
            first_t = None
            total_tok = 0
            full_text = ""

            for chunk in gateway.stream(model=req.model, messages=messages):
                if chunk:
                    if first_t is None:
                        first_t = time.time()
                    total_tok += 1
                    full_text += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            dur = round(time.time() - start_t, 3)
            ttft = round((first_t - start_t) * 1000, 1) if first_t else 0
            speed = round(total_tok / max(dur, 0.001), 1)

            term_log("🌊 [STREAM]", f"Finished in {Colors.GREEN}{dur}s{Colors.END} | TTFT: {ttft}ms | Tokens: {total_tok} | Speed: {Colors.BOLD}{speed} tok/s{Colors.END}", Colors.GREEN)
            term_log("📈 [LANGFUSE]", f"Trace synced to {settings.LANGFUSE_HOST} (Session: {req.session_id})", Colors.CYAN)
            print("-" * 80, flush=True)

            tracer.log_event(
                name="ChatStream",
                session_id=req.session_id,
                metadata={"model": req.model, "latency_s": dur, "ttft_ms": ttft, "tokens": total_tok},
                input_data=clean_prompt,
                output_data=full_text[:200]
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        t0 = time.time()
        term_log("🎙️ [VOICE INPUT]", f"Received audio: {file.filename}", Colors.BLUE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.GROQ_API_KEY)
            with open(tmp_path, "rb") as audio:
                transcript = groq.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio,
                    response_format="text"
                )
            dur = round(time.time() - t0, 3)
            text = str(transcript).strip()
            term_log("🎙️ [WHISPER TURBO]", f"Transcribed in {Colors.GREEN}{dur}s{Colors.END}: '{Colors.BOLD}{text}{Colors.END}'", Colors.GREEN)
            return JSONResponse({"text": text, "duration_s": dur})
        except Exception as e:
            term_log("❌ [WHISPER ERROR]", f"{e}", Colors.RED)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "providers": ["Groq LPU", "Google Gemini", "OpenRouter"],
            "langfuse_connected": tracer.client is not None
        }

    return app
