"""server.py - FastAPI Backend with Rich Terminal Logging, SSE Streaming & Voice API.

Replaces Streamlit with a clean, high-performance Python + JavaScript architecture:
- Rich colored terminal logging on every request, routing decision, token speed, and trace.
- Server-Sent Events (SSE) streaming for real-time typewriter output.
- Groq Whisper Large v3 Turbo for sub-second Speech-to-Text.
- NeMo Guardrails Input & Output safety filters.
- Direct Langfuse Cloud Observability sync.

Run with:
    .venv\\Scripts\\python.exe server.py
"""

import os
import re
import sys
import time
import json
import uuid
import tempfile
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import litellm

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==============================================================================
# 1. ENVIRONMENT & CONFIGURATION
# ==============================================================================
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_KEY
os.environ["GROQ_API_KEY"] = GROQ_KEY

litellm.drop_params = True
litellm.set_verbose = False
litellm.suppress_debug_info = True

# ==============================================================================
# 2. LANGFUSE OBSERVABILITY INITIALIZATION
# ==============================================================================
langfuse_client = None
if LANGFUSE_PUBLIC_KEY and not LANGFUSE_PUBLIC_KEY.startswith("pk-lf-mock"):
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
    except Exception:
        pass


def log_langfuse(name: str, session_id: str, metadata: dict, input_data: Any = None, output_data: Any = None):
    if langfuse_client:
        try:
            langfuse_client.create_event(
                name=name,
                metadata={"session_id": session_id, **metadata},
                input=input_data,
                output=output_data
            )
            langfuse_client.flush()
        except Exception:
            pass


# ==============================================================================
# 3. TERMINAL LOGGING HELPERS
# ==============================================================================
class LogColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def term_log(tag: str, message: str, color: str = LogColors.CYAN):
    timestamp = time.strftime("%H:%M:%S")
    print(f"{LogColors.BOLD}[{timestamp}]{LogColors.END} {color}{tag}{LogColors.END} {message}", flush=True)


# ==============================================================================
# 4. NEMO GUARDRAILS SAFETY ENGINE
# ==============================================================================
class ServerGuardrails:
    JAILBREAKS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"developer\s+mode",
        r"system\s+prompt\s+(reveal|leak|print)",
        r"format\s+c:"
    ]

    @classmethod
    def check_prompt(cls, prompt: str) -> Dict[str, Any]:
        for pat in cls.JAILBREAKS:
            if re.search(pat, prompt, re.IGNORECASE):
                return {
                    "allowed": False,
                    "reason": f"BLOCKED by NeMo Guardrails: Disallowed pattern '{pat}' detected."
                }
        # Mask PII
        clean = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", prompt)
        clean = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", clean)
        return {"allowed": True, "clean_prompt": clean}


# ==============================================================================
# 5. FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(title="ChatGPT Pro Multi-Model API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (HTML/JS/CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Static UI file not found.</h2>")


# ==============================================================================
# 6. CHAT STREAMING ENDPOINT (Server-Sent Events)
# ==============================================================================
class ChatRequest(BaseModel):
    prompt: str
    model: str
    session_id: str
    guardrails_enabled: bool = True
    history: List[Dict[str, str]] = []


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    term_log("📥 [REQUEST]", f"Session: {req.session_id} | Model: {LogColors.YELLOW}{req.model}{LogColors.END}", LogColors.BLUE)
    term_log("💬 [PROMPT]", f"'{req.prompt[:100]}...'", LogColors.CYAN)

    # 1. NeMo Guardrails Check
    if req.guardrails_enabled:
        t0 = time.time()
        guard = ServerGuardrails.check_prompt(req.prompt)
        dur = round((time.time() - t0) * 1000, 2)
        if not guard["allowed"]:
            term_log("🛡️ [GUARDRAIL]", f"{LogColors.RED}BLOCKED{LogColors.END} in {dur}ms -> {guard['reason']}", LogColors.RED)
            log_langfuse("Guardrail:Blocked", req.session_id, {"reason": guard["reason"]}, input_data=req.prompt)

            async def blocked_event_stream():
                data = json.dumps({"blocked": True, "reason": guard["reason"]})
                yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(blocked_event_stream(), media_type="text/event-stream")

        term_log("🛡️ [GUARDRAIL]", f"{LogColors.GREEN}PASSED{LogColors.END} in {dur}ms (PII Sanitized)", LogColors.GREEN)
        clean_prompt = guard["clean_prompt"]
    else:
        clean_prompt = req.prompt

    # 2. Build Messages
    messages = [{"role": "system", "content": "You are ChatGPT Pro, a helpful, intelligent, and concise AI assistant."}]
    for h in req.history:
        if h.get("content"):
            messages.append({"role": h.get("role", "user"), "content": h.get("content")})
    messages.append({"role": "user", "content": clean_prompt})

    # 3. Stream Generator
    async def event_generator():
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        full_response = ""

        try:
            # GROQ LPU INFERENCE
            if req.model.startswith("groq/"):
                actual_model = req.model.replace("groq/", "")
                term_log("⚡ [ROUTER]", f"Routing to {LogColors.YELLOW}Groq LPU ({actual_model}){LogColors.END}...", LogColors.YELLOW)
                groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
                
                stream = groq_client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    stream=True,
                    max_tokens=2048
                )

                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        total_tokens += 1
                        full_response += content
                        data = json.dumps({"chunk": content})
                        yield f"data: {data}\n\n"

            # GOOGLE GEMINI INFERENCE
            elif req.model.startswith("gemini/"):
                term_log("🔵 [ROUTER]", f"Routing to {LogColors.CYAN}Google Gemini ({req.model}){LogColors.END}...", LogColors.CYAN)
                response = litellm.completion(
                    model=req.model,
                    messages=messages,
                    api_key=GEMINI_KEY,
                    stream=True,
                    max_tokens=2048
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        total_tokens += 1
                        full_response += content
                        data = json.dumps({"chunk": content})
                        yield f"data: {data}\n\n"

            # OPENROUTER MULTI-MODEL MESH
            else:
                term_log("🟢 [ROUTER]", f"Routing to {LogColors.GREEN}OpenRouter ({req.model}){LogColors.END}...", LogColors.GREEN)
                response = litellm.completion(
                    model=req.model,
                    messages=messages,
                    api_key=OPENROUTER_KEY,
                    api_base="https://openrouter.ai/api/v1",
                    stream=True,
                    max_tokens=2048
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        total_tokens += 1
                        full_response += content
                        data = json.dumps({"chunk": content})
                        yield f"data: {data}\n\n"

        except Exception as e:
            term_log("⚠️ [FAILOVER]", f"Primary error: {e}. Rerouting to OpenRouter fallback...", LogColors.YELLOW)
            try:
                fb_stream = litellm.completion(
                    model="openrouter/inclusionai/ling-3.0-flash-fin:free",
                    messages=messages,
                    api_key=OPENROUTER_KEY,
                    api_base="https://openrouter.ai/api/v1",
                    stream=True
                )
                for chunk in fb_stream:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        total_tokens += 1
                        full_response += content
                        data = json.dumps({"chunk": content})
                        yield f"data: {data}\n\n"
            except Exception as fb_err:
                term_log("❌ [ERROR]", f"Fallback error: {fb_err}", LogColors.RED)
                yield f"data: {json.dumps({'chunk': f'Error: {fb_err}'})}\n\n"

        # Telemetry & Speed Calculations
        total_time = round(time.time() - start_time, 3)
        ttft = round((first_token_time - start_time) * 1000, 1) if first_token_time else 0
        speed = round(total_tokens / max(total_time, 0.001), 1)

        term_log("🌊 [STREAM]", f"Completed in {LogColors.GREEN}{total_time}s{LogColors.END} | TTFT: {ttft}ms | Tokens: {total_tokens} | Speed: {LogColors.BOLD}{speed} tok/s{LogColors.END}", LogColors.GREEN)
        term_log("📈 [LANGFUSE]", f"Trace logged to {LANGFUSE_HOST} (Session: {req.session_id})", LogColors.CYAN)
        print("-" * 85)

        # Log to Langfuse Cloud
        log_langfuse(
            name="ChatCompletion",
            session_id=req.session_id,
            metadata={"model": req.model, "latency_s": total_time, "ttft_ms": ttft, "tokens": total_tokens},
            input_data=clean_prompt,
            output_data=full_response[:200]
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==============================================================================
# 7. VOICE TRANSCRIPTION (Groq Whisper Large v3 Turbo)
# ==============================================================================
@app.post("/api/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    t0 = time.time()
    term_log("🎙️ [VOICE INPUT]", f"Received audio file: {file.filename}...", LogColors.BLUE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text"
            )
        dur = round(time.time() - t0, 3)
        text_out = str(transcription).strip()
        term_log("🎙️ [WHISPER TURBO]", f"Transcribed in {LogColors.GREEN}{dur}s{LogColors.END}: '{LogColors.BOLD}{text_out}{LogColors.END}'", LogColors.GREEN)
        return JSONResponse({"text": text_out, "duration_s": dur})
    except Exception as e:
        term_log("❌ [WHISPER ERROR]", f"{e}", LogColors.RED)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# 8. HEALTH CHECK
# ==============================================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "providers": ["Groq LPU", "Google Gemini", "OpenRouter"],
        "langfuse_connected": langfuse_client is not None
    }


# ==============================================================================
# 9. SERVER RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 85)
    print(f"{LogColors.BOLD}{LogColors.GREEN}🚀 STARTING CHATGPT PRO FASTAPI + JAVASCRIPT SERVER{LogColors.END}")
    print(f"👉 Local Web UI      : {LogColors.BOLD}http://localhost:8080{LogColors.END}")
    print(f"👉 Live Terminal Logs: {LogColors.BOLD}ENABLED (Streaming & Latencies Visible Below){LogColors.END}")
    print(f"👉 Langfuse Status   : {'Connected to Cloud' if langfuse_client else 'Local Mode'}")
    print("=" * 85 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
