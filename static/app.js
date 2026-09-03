/**
 * ChatGPT Pro Frontend Application Logic
 * 
 * Manages:
 *  1. Server-Sent Events (SSE) Token Streaming for real-time typewriter output.
 *  2. Voice Mode: MediaRecorder audio capture with Groq Whisper Turbo STT.
 *  3. Voice Output: Web SpeechSynthesis API for auto-spoken responses.
 *  4. Markdown & Code Rendering: Marked.js + Highlight.js with 1-click Copy Code buttons.
 *  5. Chat History & Session Tracking.
 */

// ============================================================================
// 1. Application State & Session Configuration
// ============================================================================
let currentSessionId = 'chat-' + Math.random().toString(36).substring(2, 9);
let chatHistory = [];
let isStreaming = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// ============================================================================
// 2. DOM Elements
// ============================================================================
const chatContainer = document.getElementById('chatContainer');
const starterCards = document.getElementById('starterCards');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const voiceOrb = document.getElementById('voiceOrb');
const voiceStatus = document.getElementById('voiceStatus');
const modelSelect = document.getElementById('modelSelect');
const guardrailToggle = document.getElementById('guardrailToggle');
const ttsToggle = document.getElementById('ttsToggle');
const newChatBtn = document.getElementById('newChatBtn');

// Auto-adjust textarea height dynamically as the user types
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
});

// Submit on Enter key (Shift+Enter for new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', startNewChat);

// ============================================================================
// 3. New Chat & Starter Prompts
// ============================================================================
function startNewChat() {
    chatHistory = [];
    currentSessionId = 'chat-' + Math.random().toString(36).substring(2, 9);
    chatContainer.innerHTML = '';
    starterCards.style.display = 'grid';
    messageInput.value = '';
    messageInput.style.height = 'auto';
    window.speechSynthesis.cancel();
}

function sendStarter(text) {
    messageInput.value = text;
    sendMessage();
}

// ============================================================================
// 4. Message Dispatch & SSE Streaming Handler
// ============================================================================
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isStreaming) return;

    // Hide welcome starter cards on first message
    starterCards.style.display = 'none';

    // 1. Render User Message
    appendMessage('user', text);
    chatHistory.push({ role: 'user', content: text });
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // 2. Render Assistant Placeholder Bubble with Typing Pulse
    const botMsgDiv = appendMessage('assistant', '', true);
    const contentDiv = botMsgDiv.querySelector('.msg-content');

    isStreaming = true;
    sendBtn.disabled = true;

    let fullResponse = '';
    const selectedModel = modelSelect.value;
    const guardrailsActive = guardrailToggle.checked;

    try {
        // Dispatch POST request to FastAPI SSE endpoint
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: text,
                model: selectedModel,
                session_id: currentSessionId,
                guardrails_enabled: guardrailsActive,
                history: chatHistory.slice(-6) // Send up to last 6 turns for context
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        // Read Server-Sent Events stream chunk-by-chunk
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Retain incomplete trailing line in buffer

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data: ')) continue;
                
                const dataStr = trimmed.replace('data: ', '').trim();
                if (dataStr === '[DONE]') break;

                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.blocked) {
                        fullResponse = `🛡️ **[Blocked by NeMo Guardrails]**\n\n*${parsed.reason}*`;
                        renderMarkdown(contentDiv, fullResponse);
                        break;
                    }
                    if (parsed.chunk) {
                        fullResponse += parsed.chunk;
                        renderMarkdown(contentDiv, fullResponse);
                        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    }
                } catch (jsonErr) {
                    // Ignore transient chunk framing errors
                }
            }
        }

        chatHistory.push({ role: 'assistant', content: fullResponse });

        // Auto-speak response if TTS toggle is active
        if (ttsToggle.checked && fullResponse) {
            speakText(fullResponse);
        }

    } catch (err) {
        contentDiv.innerHTML = `<span class="text-red-400">❌ Error connecting to server: ${err.message}</span>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        // Remove typing pulse indicator
        const pulse = botMsgDiv.querySelector('.typing-pulse');
        if (pulse) pulse.remove();
    }
}

// ============================================================================
// 5. DOM Rendering & Markdown Formatting
// ============================================================================
function appendMessage(role, text, isPending = false) {
    const wrapper = document.createElement('div');
    wrapper.className = `flex gap-4 p-4 rounded-xl ${role === 'user' ? 'bg-[#212121]' : 'bg-transparent'}`;

    const avatar = document.createElement('div');
    avatar.className = `w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shrink-0 ${
        role === 'user' ? 'bg-purple-600 text-white' : 'bg-emerald-600 text-white'
    }`;
    avatar.innerText = role === 'user' ? 'U' : 'AI';

    const body = document.createElement('div');
    body.className = 'flex-1 overflow-hidden';

    const author = document.createElement('div');
    author.className = 'font-semibold text-xs text-gray-400 mb-1';
    author.innerText = role === 'user' ? 'You' : 'ChatGPT Pro';

    const content = document.createElement('div');
    content.className = 'msg-content text-gray-200 text-sm leading-relaxed';

    if (isPending) {
        content.innerHTML = '<span class="typing-pulse"></span>';
    } else {
        renderMarkdown(content, text);
    }

    body.appendChild(author);
    body.appendChild(content);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);

    chatContainer.appendChild(wrapper);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

    return wrapper;
}

function renderMarkdown(element, markdownText) {
    // Configure Marked.js options
    marked.setOptions({
        highlight: function(code, lang) {
            const language = highlight.getLanguage(lang) ? lang : 'plaintext';
            return highlight.highlight(code, { language }).value;
        },
        breaks: true,
        gfm: true
    });

    element.innerHTML = marked.parse(markdownText);

    // Attach 1-click copy buttons to code blocks
    element.querySelectorAll('pre').forEach((pre) => {
        if (!pre.querySelector('.copy-code-btn')) {
            const btn = document.createElement('button');
            btn.className = 'copy-code-btn';
            btn.innerText = 'Copy code';
            btn.onclick = () => {
                const code = pre.querySelector('code')?.innerText || pre.innerText;
                navigator.clipboard.writeText(code);
                btn.innerText = '✓ Copied!';
                setTimeout(() => { btn.innerText = 'Copy code'; }, 2000);
            };
            pre.style.position = 'relative';
            pre.appendChild(btn);
        }
    });
}

// ============================================================================
// 6. Voice Mode: MediaRecorder STT & Web SpeechSynthesis TTS
// ============================================================================
voiceBtn.addEventListener('click', toggleVoiceRecording);

async function toggleVoiceRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioForTranscription(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        voiceBtn.classList.add('bg-red-600', 'recording-active');
        voiceOrb.classList.remove('hidden');
        voiceStatus.innerText = 'Listening... Speak now';
    } catch (err) {
        alert('Microphone access denied or unavailable: ' + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        voiceBtn.classList.remove('bg-red-600', 'recording-active');
        voiceOrb.classList.add('hidden');
        voiceStatus.innerText = 'Transcribing audio with Groq Whisper Turbo...';
    }
}

async function sendAudioForTranscription(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'speech.wav');

    try {
        const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`Transcription HTTP error: ${res.status}`);
        const data = await res.json();

        if (data.text) {
            messageInput.value = data.text;
            voiceStatus.innerText = `Transcribed in ${data.duration_s}s`;
            sendMessage();
        } else {
            voiceStatus.innerText = 'No speech recognized.';
        }
    } catch (e) {
        voiceStatus.innerText = 'Voice transcription error: ' + e.message;
    }
}

function speakText(text) {
    // Strip markdown formatting characters for natural speech synthesis
    const cleanText = text.replace(/[*#`_~\[\]]/g, '').trim();
    if (!cleanText) return;

    window.speechSynthesis.cancel(); // Stop any currently playing audio
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}
