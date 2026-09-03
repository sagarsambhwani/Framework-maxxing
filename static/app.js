/**
 * app.js - Modern ChatGPT Client with Real-Time Streaming & Voice Mode
 */

// 1. STATE & VARIABLES
let currentSessionId = 'chat-' + Math.random().toString(36).substring(2, 9);
let conversationHistory = [];
let isStreaming = false;
let abortController = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// DOM Elements
const messagesContainer = document.getElementById('messages-container');
const starterCardsContainer = document.getElementById('starter-cards-container');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const modelSelect = document.getElementById('model-select');
const newChatBtn = document.getElementById('new-chat-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const autoTtsToggle = document.getElementById('auto-tts-toggle');
const guardrailsToggle = document.getElementById('guardrails-toggle');
const headerModelBadge = document.getElementById('header-model-badge');
const headerLatencyBadge = document.getElementById('header-latency-badge');
const activeSessionPill = document.getElementById('active-session-pill');
const typingIndicator = document.getElementById('typing-indicator');
const voiceIndicator = document.getElementById('voice-indicator');
const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
const sidebar = document.getElementById('sidebar');

// Configure marked.js for syntax highlighting
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true
});

// Update active session pill
activeSessionPill.textContent = '#' + currentSessionId.substring(0, 10);

// 2. MODEL BADGE UPDATE
const MODEL_BADGES = {
    'groq/qwen/qwen3.8-27b': { name: 'Qwen 3.8 27B', latency: '⚡ LPU 0.3s' },
    'groq/groq/compound': { name: 'Groq Compound', latency: '🧠 Reasoning' },
    'groq/openai/gpt-oss-120b': { name: 'GPT-OSS 120B', latency: '🚀 LPU 0.5s' },
    'groq/allam-2-7b': { name: 'Allam 7B', latency: '⚡ 110ms Ultra' },
    'gemini/gemma-4-31b-it': { name: 'Gemma 4 31B', latency: '🔵 Google AI' },
    'gemini/gemini-3.6-flash': { name: 'Gemini 3.6 Flash', latency: '🌊 1M Context' },
    'openrouter/inclusionai/ling-3.0-flash-fin:free': { name: 'Ling 3.0 Flash', latency: '🟢 OpenRouter' },
    'openrouter/nvidia/nemotron-3.5-lightning:free': { name: 'Nemotron 3.5', latency: '🟢 OpenRouter' }
};

function updateModelHeader() {
    const selected = modelSelect.value;
    const meta = MODEL_BADGES[selected] || { name: selected, latency: '⚡ Active' };
    headerModelBadge.textContent = meta.name;
    headerLatencyBadge.textContent = meta.latency;
}
modelSelect.addEventListener('change', updateModelHeader);
updateModelHeader();

// 3. AUTO-RESIZE INPUT TEXTAREA
promptInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

promptInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
});

// 4. STARTER CARDS CLICK
document.querySelectorAll('.starter-card').forEach(card => {
    card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        promptInput.value = prompt;
        handleSendMessage();
    });
});

// 5. NEW CHAT & CLEAR
newChatBtn.addEventListener('click', () => {
    currentSessionId = 'chat-' + Math.random().toString(36).substring(2, 9);
    activeSessionPill.textContent = '#' + currentSessionId.substring(0, 10);
    conversationHistory = [];
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(starterCardsContainer);
    starterCardsContainer.style.display = 'flex';
    promptInput.value = '';
    promptInput.focus();
});

clearChatBtn.addEventListener('click', () => {
    if (confirm('Clear current chat conversation?')) {
        conversationHistory = [];
        messagesContainer.innerHTML = '';
        messagesContainer.appendChild(starterCardsContainer);
        starterCardsContainer.style.display = 'flex';
    }
});

// 6. TOGGLE SIDEBAR (MOBILE/COLLAPSE)
toggleSidebarBtn.addEventListener('click', () => {
    sidebar.classList.toggle('-ml-64');
});

// 7. SEND MESSAGE & STREAMING HANDLER
async function handleSendMessage() {
    const text = promptInput.value.trim();
    if (!text || isStreaming) return;

    // Hide starter cards if visible
    if (starterCardsContainer) {
        starterCardsContainer.style.display = 'none';
    }

    // Reset input
    promptInput.value = '';
    promptInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Append User Message to UI
    appendMessage('user', text);
    conversationHistory.push({ role: 'user', content: text });

    // Create Assistant Message Placeholder
    const assistantMsgObj = createAssistantPlaceholder();
    const contentDiv = assistantMsgObj.contentDiv;
    const telemetryDiv = assistantMsgObj.telemetryDiv;

    isStreaming = true;
    typingIndicator.classList.remove('hidden');
    typingIndicator.classList.add('flex');
    contentDiv.classList.add('cursor-pulse');

    const startTime = performance.now();
    let accumulatedText = '';
    let tokenCount = 0;

    abortController = new AbortController();

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: text,
                model: modelSelect.value,
                session_id: currentSessionId,
                guardrails_enabled: guardrailsToggle.checked,
                history: conversationHistory.slice(-8)
            }),
            signal: abortController.signal
        });

        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.replace('data: ', '').trim();
                    if (dataStr === '[DONE]') break;

                    try {
                        const parsed = JSON.parse(dataStr);
                        if (parsed.chunk) {
                            accumulatedText += parsed.chunk;
                            tokenCount++;
                            contentDiv.innerHTML = marked.parse(accumulatedText);
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        } else if (parsed.blocked) {
                            accumulatedText = `❌ **[Request Blocked by NeMo Guardrails]**\n\n${parsed.reason}`;
                            contentDiv.innerHTML = marked.parse(accumulatedText);
                        }
                    } catch (err) {
                        // Fallback plain text chunk
                        accumulatedText += dataStr;
                        contentDiv.innerHTML = marked.parse(accumulatedText);
                    }
                }
            }
        }

        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        const meta = MODEL_BADGES[modelSelect.value] || { name: modelSelect.value };
        
        telemetryDiv.innerHTML = `
            <div class="inline-flex items-center gap-2 text-[11px] text-gray-400 bg-chatSidebar px-2.5 py-1 rounded-full border border-chatBorder mt-2">
                <span>⚡ <b>${duration}s</b></span> • 
                <span>🤖 ${meta.name}</span> • 
                <span>🛡️ ${guardrailsToggle.checked ? 'NeMo Guardrails Active' : 'Off'}</span>
            </div>
        `;

        conversationHistory.push({ role: 'assistant', content: accumulatedText });

        // Auto Text-to-Speech if enabled
        if (autoTtsToggle.checked && accumulatedText) {
            speakText(accumulatedText);
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            contentDiv.innerHTML += '<p class="text-xs text-yellow-400 mt-2"><i>[Generation stopped by user]</i></p>';
        } else {
            contentDiv.innerHTML = `<p class="text-xs text-red-400 mt-2">❌ Error: ${err.message}</p>`;
        }
    } finally {
        contentDiv.classList.remove('cursor-pulse');
        isStreaming = false;
        sendBtn.disabled = false;
        typingIndicator.classList.add('hidden');
        typingIndicator.classList.remove('flex');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// 8. DOM MESSAGE HELPERS
function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex gap-3 text-sm ' + (role === 'user' ? 'justify-end' : 'justify-start');

    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="max-w-[80%] bg-chatInput border border-chatBorder text-gray-100 rounded-2xl px-4 py-3 shadow-md">
                ${escapeHtml(text)}
            </div>
            <div class="w-7 h-7 rounded-full bg-chatBorder flex items-center justify-center text-xs text-gray-300 font-bold shrink-0">
                <i class="fa-solid fa-user"></i>
            </div>
        `;
    }
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createAssistantPlaceholder() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex gap-3 text-sm justify-start w-full';

    const id = 'assistant-content-' + Date.now();
    const telId = 'assistant-tel-' + Date.now();

    msgDiv.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-chatAccent flex items-center justify-center text-xs text-white font-bold shrink-0 shadow-md">
            <i class="fa-solid fa-bolt"></i>
        </div>
        <div class="flex-1 max-w-[85%]">
            <div id="${id}" class="text-gray-200 leading-relaxed space-y-2"></div>
            <div id="${telId}"></div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return {
        contentDiv: document.getElementById(id),
        telemetryDiv: document.getElementById(telId)
    };
}

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// 9. VOICE MODE: SPEECH-TO-TEXT (MICROPHONE RECORDING)
micBtn.addEventListener('click', async () => {
    if (isRecording) {
        stopAudioRecording();
    } else {
        startAudioRecording();
    }
});

async function startAudioRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            stream.getTracks().forEach(track => track.stop());
            await sendAudioToTranscribe(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add('recording-pulse');
        voiceIndicator.classList.remove('hidden');
        voiceIndicator.classList.add('flex');
    } catch (err) {
        alert('Microphone access denied or not available: ' + err.message);
    }
}

function stopAudioRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove('recording-pulse');
        voiceIndicator.classList.add('hidden');
        voiceIndicator.classList.remove('flex');
    }
}

async function sendAudioToTranscribe(audioBlob) {
    typingIndicator.classList.remove('hidden');
    typingIndicator.classList.add('flex');
    typingIndicator.innerHTML = '<i class="fa-solid fa-microphone-lines animate-pulse text-xs"></i> Transcribing with Groq Whisper Turbo...';

    const formData = new FormData();
    formData.append('file', audioBlob, 'speech.wav');

    try {
        const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.text) {
            promptInput.value = data.text;
            handleSendMessage();
        }
    } catch (e) {
        alert('Transcription failed: ' + e.message);
    } finally {
        typingIndicator.innerHTML = '<i class="fa-solid fa-circle-notch animate-spin text-xs"></i> Generating tokens on terminal...';
        typingIndicator.classList.add('hidden');
        typingIndicator.classList.remove('flex');
    }
}

// 10. VOICE MODE: TEXT-TO-SPEECH (BROWSER TTS)
function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    // Clean markdown and code out of speech text
    const cleanText = text
        .replace(/```[\s\S]*?```/g, 'Code snippet omitted.')
        .replace(/[*#_`]/g, '')
        .substring(0, 400);

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

// Attach Send Button
sendBtn.addEventListener('click', handleSendMessage);
