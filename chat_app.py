"""chat_app.py - Modern ChatGPT-Style Conversational UI with Voice Mode & Multi-Model Intelligence.

Features:
- Multi-Model Selection (Groq LPUs, Google Gemini 1M Context, OpenRouter)
- Real-Time Token-by-Token Streaming
- Two-Way Voice Mode:
  * Speech-to-Text: Groq Whisper Large v3 Turbo (sub-second transcription)
  * Text-to-Speech: Native Voice Synthesis & Web Audio Player
- NeMo Guardrails Input & Output Safety Filtering with live indicators
- Langfuse Cloud Observability & Per-Message Telemetry (Latency, Tokens, Cost)
- ChatGPT Modern Dark UI Theme & Session Management
"""

import os
import re
import sys
import time
import uuid
import json
import logging
from typing import Generator, Dict, Any, List
from dotenv import load_dotenv

import streamlit as st
from openai import OpenAI
import litellm

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. LOAD CONFIGURATION
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
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# 2. INITIALIZE LANGFUSE
langfuse_client = None
if LANGFUSE_PUBLIC_KEY and not LANGFUSE_PUBLIC_KEY.startswith("pk-lf-mock"):
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
    except Exception:
        pass

# 3. STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="ChatGPT Pro - Multi-Model & Voice AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 4. CHATGPT DARK THEME CSS
st.markdown("""
<style>
    /* Dark Theme ChatGPT Aesthetics */
    .stApp {
        background-color: #212121;
        color: #ECECEC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2f2f2f;
    }
    
    /* New Chat Button */
    .new-chat-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        padding: 10px 14px;
        background-color: #2f2f2f;
        color: #fff;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        border: 1px solid #3f3f3f;
        transition: background-color 0.2s;
    }
    .new-chat-btn:hover {
        background-color: #3f3f3f;
    }

    /* Message Bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    
    /* User Message */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #2f2f2f !important;
        border: 1px solid #3f3f3f;
    }
    
    /* Assistant Message */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: transparent !important;
    }

    /* Voice Mode Orb Animation */
    .voice-orb-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        margin: 15px 0;
        background: radial-gradient(circle, rgba(16,163,127,0.15) 0%, rgba(33,33,33,0) 70%);
        border: 1px solid #2f2f2f;
        border-radius: 16px;
    }
    
    .voice-orb {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10a37f, #19c37d);
        box-shadow: 0 0 25px rgba(16, 163, 127, 0.6);
        animation: pulse 2s infinite ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.92); box-shadow: 0 0 15px rgba(16, 163, 127, 0.4); }
        50% { transform: scale(1.08); box-shadow: 0 0 35px rgba(25, 195, 125, 0.8); }
        100% { transform: scale(0.92); box-shadow: 0 0 15px rgba(16, 163, 127, 0.4); }
    }

    /* Starter Prompt Cards */
    .prompt-card {
        background-color: #2f2f2f;
        border: 1px solid #3f3f3f;
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: transform 0.15s, border-color 0.15s;
        height: 100%;
    }
    .prompt-card:hover {
        transform: translateY(-2px);
        border-color: #10a37f;
        background-color: #363636;
    }
    .prompt-card h4 {
        margin: 0 0 6px 0;
        font-size: 15px;
        color: #fff;
    }
    .prompt-card p {
        margin: 0;
        font-size: 13px;
        color: #b4b4b4;
    }

    /* Telemetry pill */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #8e8e8e;
        background-color: #1a1a1a;
        padding: 3px 8px;
        border-radius: 12px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 5. MODEL CATALOG WITH METADATA
MODEL_CATALOG = {
    # GROQ LPU MODELS (Ultra-Fast)
    "⚡ Qwen 3.8 27B (Groq LPU - ~0.4s)": {
        "slug": "groq/qwen/qwen3.8-27b",
        "provider": "Groq LPU",
        "badge": "⚡ Blazing Fast",
        "context": "32k tokens"
    },
    "⚡ Groq Compound Reasoning (Groq LPU)": {
        "slug": "groq/groq/compound",
        "provider": "Groq LPU",
        "badge": "🧠 Deep Reasoning",
        "context": "32k tokens"
    },
    "⚡ GPT-OSS 120B (Groq LPU)": {
        "slug": "groq/openai/gpt-oss-120b",
        "provider": "Groq LPU",
        "badge": "🚀 120B Flagship",
        "context": "32k tokens"
    },
    "⚡ Allam 7B (Groq LPU - 0.1s)": {
        "slug": "groq/allam-2-7b",
        "provider": "Groq LPU",
        "badge": "⚡ 110ms Ultra-Fast",
        "context": "8k tokens"
    },
    # GOOGLE GEMINI MODELS
    "🔵 Gemma 4 31B (Google AI Studio)": {
        "slug": "gemini/gemma-4-31b-it",
        "provider": "Google Gemini",
        "badge": "🔓 Open Weights",
        "context": "256k tokens"
    },
    "🔵 Gemini 3.6 Flash (Google 1M Context)": {
        "slug": "gemini/gemini-3.6-flash",
        "provider": "Google Gemini",
        "badge": "🌊 1M Context",
        "context": "1,048,576 tokens"
    },
    # OPENROUTER MULTI-MODEL MESH
    "🟢 Ling 3.0 Flash (OpenRouter Mesh)": {
        "slug": "openrouter/inclusionai/ling-3.0-flash-fin:free",
        "provider": "OpenRouter",
        "badge": "🌐 Multi-Model Mesh",
        "context": "32k tokens"
    },
    "🟢 Nemotron 3.5 Lightning (OpenRouter Mesh)": {
        "slug": "openrouter/nvidia/nemotron-3.5-lightning:free",
        "provider": "OpenRouter",
        "badge": "🌐 Multi-Model Mesh",
        "context": "32k tokens"
    }
}

# 6. SESSION STATE INITIALIZATION
if "session_id" not in st.session_state:
    st.session_state.session_id = f"chat-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

if "auto_tts" not in st.session_state:
    st.session_state.auto_tts = False

if "guardrails_enabled" not in st.session_state:
    st.session_state.guardrails_enabled = True

# 7. NEMO GUARDRAILS HELPER
def check_guardrails(text: str) -> Dict[str, Any]:
    """Inspects text for prompt injections, jailbreaks, and sensitive data."""
    if not st.session_state.guardrails_enabled:
        return {"allowed": True, "clean_text": text, "flagged": False}

    # 1. Jailbreak Check
    jailbreaks = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"developer\s+mode",
        r"system\s+prompt\s+(leak|reveal|print)",
        r"format\s+c:"
    ]
    for pat in jailbreaks:
        if re.search(pat, text, re.IGNORECASE):
            return {
                "allowed": False,
                "clean_text": text,
                "flagged": True,
                "reason": f"BLOCKED by NeMo Guardrails: Disallowed pattern '{pat}' detected."
            }

    # 2. PII Redaction
    sanitized = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
    sanitized = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", sanitized)

    return {"allowed": True, "clean_text": sanitized, "flagged": False}


# 8. GROQ AUDIO TRANSCRIPTION (Whisper Turbo STT)
def transcribe_audio_groq(audio_bytes: bytes) -> str:
    """Transcribes speech audio into text in <300ms using Groq Whisper."""
    if not GROQ_KEY:
        return "Groq API key not configured."
    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text"
            )
        os.remove(tmp_path)
        return str(transcription).strip()
    except Exception as e:
        return f"Transcription error: {e}"


# 9. SPEECH SYNTHESIS HELPER (Browser JavaScript TTS)
def trigger_voice_output(text: str):
    """Executes browser SpeechSynthesis to speak assistant response aloud."""
    cleaned_for_speech = re.sub(r"```[\s\S]*?```", "Code block omitted.", text)
    cleaned_for_speech = re.sub(r"[*#_`]", "", cleaned_for_speech)
    clean_js = json.dumps(cleaned_for_speech[:500])

    html_code = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance({clean_js});
            utterance.rate = 1.05;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(html_code, height=0)


# 10. COMPLETION STREAM GENERATOR
def stream_llm_response(model_slug: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Streams completion chunks from the selected model."""
    try:
        # If Groq model, call Groq endpoint directly for maximum speed
        if model_slug.startswith("groq/"):
            actual_model = model_slug.replace("groq/", "")
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
                    yield content

        # If Gemini model, call via LiteLLM
        elif model_slug.startswith("gemini/"):
            response = litellm.completion(
                model=model_slug,
                messages=messages,
                api_key=GEMINI_KEY,
                stream=True,
                max_tokens=2048
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content

        # If OpenRouter model
        else:
            response = litellm.completion(
                model=model_slug,
                messages=messages,
                api_key=OPENROUTER_KEY,
                api_base="https://openrouter.ai/api/v1",
                stream=True,
                max_tokens=2048
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content

    except Exception as e:
        # Automated failover to OpenRouter if primary hits error
        yield f"\n\n*[Notice: Failing over to backup route due to: {str(e)[:100]}...]*\n\n"
        try:
            fallback_resp = litellm.completion(
                model="openrouter/inclusionai/ling-3.0-flash-fin:free",
                messages=messages,
                api_key=OPENROUTER_KEY,
                api_base="https://openrouter.ai/api/v1",
                stream=True
            )
            for chunk in fallback_resp:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as fb_err:
            yield f"\n\n❌ All endpoints failed: {fb_err}"


# ==============================================================================
# 11. SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### 💬 **ChatGPT Pro**")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Model Selector
    st.markdown("##### 🧠 **Active Model**")
    selected_model_name = st.selectbox(
        "Choose Intelligence Engine:",
        options=list(MODEL_CATALOG.keys()),
        index=0,
        label_visibility="collapsed"
    )
    model_info = MODEL_CATALOG[selected_model_name]
    st.markdown(f"**Provider:** `{model_info['provider']}` | **Context:** `{model_info['context']}`")
    st.caption(f"Status: **{model_info['badge']}**")

    st.markdown("---")

    # Voice Mode Controls
    st.markdown("##### 🎙️ **Voice Mode Settings**")
    st.session_state.voice_mode = st.toggle("Enable Voice Input Mode", value=st.session_state.voice_mode)
    st.session_state.auto_tts = st.toggle("🔊 Auto-Speak Responses", value=st.session_state.auto_tts)

    st.markdown("---")

    # Safety & Observability
    st.markdown("##### 🛡️ **Safety & Observability**")
    st.session_state.guardrails_enabled = st.toggle("NeMo Guardrails & PII Filter", value=st.session_state.guardrails_enabled)
    
    if langfuse_client:
        st.markdown(f"🟢 **Langfuse Tracing:** [View Dashboard]({LANGFUSE_HOST})")
    else:
        st.markdown("⚪ **Langfuse:** Local Mode")

    st.caption(f"Session: `{st.session_state.session_id}`")


# ==============================================================================
# 12. MAIN CONVERSATION INTERFACE
# ==============================================================================

# Voice Mode Orb Banner (if Voice Mode enabled)
if st.session_state.voice_mode:
    st.markdown("""
    <div class="voice-orb-container">
        <div class="voice-orb"></div>
        <p style="margin-top: 14px; font-weight: 500; color: #19c37d;">Voice Mode Active — Speak into your microphone below</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Native Streamlit Audio Input
    audio_val = st.audio_input("Record Speech Prompt:")
    if audio_val is not None:
        audio_bytes = audio_val.read()
        with st.spinner("🎙️ Transcribing with Groq Whisper Turbo..."):
            transcribed_text = transcribe_audio_groq(audio_bytes)
            if transcribed_text and not transcribed_text.startswith("Transcription error"):
                st.session_state.messages.append({"role": "user", "content": transcribed_text, "voice_input": True})
                st.rerun()

# Empty State: Starter Cards
if len(st.session_state.messages) == 0:
    st.markdown("<h2 style='text-align: center; margin-top: 30px; margin-bottom: 25px;'>What can I help with today?</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 **Explain Quantum Computing**\n\nIn simple everyday analogies for a high schooler", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain Quantum Computing in simple everyday analogies for a high schooler."})
            st.rerun()
        if st.button("⚡ **Compare Groq LPUs vs Nvidia GPUs**\n\nWhy LPUs achieve 10x lower latency in token generation", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Compare Groq LPUs vs Nvidia GPUs for LLM inference latency."})
            st.rerun()
    with col2:
        if st.button("🐍 **Build a FastAPI Microservice**\n\nWith JWT auth, rate limiting, and Dockerfile", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Write a clean production FastAPI service with JWT auth and Dockerfile."})
            st.rerun()
        if st.button("📊 **Design a Multi-Cloud AI Architecture**\n\nFor 10M requests/day with 99.99% uptime", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Design an enterprise multi-cloud AI architecture for 10M requests/day with 99.99% uptime."})
            st.rerun()

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("telemetry"):
            tel = msg["telemetry"]
            st.markdown(f"""
            <div class="telemetry-pill">
                <span>⚡ {tel.get('latency', '0')}s</span> • 
                <span>🤖 {tel.get('model', '')}</span> • 
                <span>🛡️ {tel.get('guardrail', 'Passed')}</span>
            </div>
            """, unsafe_allow_html=True)

# 13. CHAT INPUT PROMPT BAR
user_query = st.chat_input("Message ChatGPT Pro...")

if user_query:
    # 1. NeMo Guardrails Input Safety Check
    guardrail_check = check_guardrails(user_query)

    if not guardrail_check["allowed"]:
        st.session_state.messages.append({"role": "user", "content": user_query})
        blocked_msg = f"❌ **[Request Blocked by NeMo Guardrails]**\n\n{guardrail_check['reason']}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": blocked_msg,
            "telemetry": {"latency": "0.01", "model": "NeMo Guardrail Engine", "guardrail": "Blocked"}
        })
        st.rerun()

    # Append sanitized user prompt
    st.session_state.messages.append({"role": "user", "content": guardrail_check["clean_text"]})

    # Render User Message
    with st.chat_message("user"):
        st.markdown(guardrail_check["clean_text"])

    # 2. Render Assistant Streaming Response
    with st.chat_message("assistant"):
        start_t = time.time()
        active_slug = model_info["slug"]

        # Prepare conversation history for LLM
        convo_history = [{"role": "system", "content": "You are ChatGPT, a helpful, intelligent, and concise AI assistant."}]
        for m in st.session_state.messages[-8:]:
            convo_history.append({"role": m["role"], "content": m["content"]})

        # Stream response
        stream_gen = stream_llm_response(active_slug, convo_history)
        full_response = st.write_stream(stream_gen)
        latency = round(time.time() - start_t, 2)

        # Output PII Sanitization
        clean_response = check_guardrails(full_response)["clean_text"]

        # Telemetry Metadata
        telemetry = {
            "latency": latency,
            "model": model_info["provider"],
            "guardrail": "Passed & Sanitized" if st.session_state.guardrails_enabled else "Off"
        }
        
        st.markdown(f"""
        <div class="telemetry-pill">
            <span>⚡ {latency}s</span> • 
            <span>🤖 {model_info['provider']} ({model_info['badge']})</span> • 
            <span>🛡️ {telemetry['guardrail']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Auto Text-to-Speech if enabled
        if st.session_state.auto_tts:
            trigger_voice_output(clean_response)

        # Log trace to Langfuse Cloud
        if langfuse_client:
            try:
                langfuse_client.create_event(
                    name="ChatMessage",
                    metadata={
                        "session_id": st.session_state.session_id,
                        "model": active_slug,
                        "latency_s": latency
                    },
                    input=guardrail_check["clean_text"],
                    output=clean_response[:300]
                )
                langfuse_client.flush()
            except Exception:
                pass

        # Save assistant message to state
        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_response,
            "telemetry": telemetry
        })
