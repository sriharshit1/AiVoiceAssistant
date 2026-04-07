/* ═══════════════════════════════════════════════════════════
   app.js — ARIA Voice Assistant frontend logic
═══════════════════════════════════════════════════════════ */

const API    = window.location.origin;
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

// ── State ──────────────────────────────────────────────────
let ws;
let mediaRecorder;
let audioChunks    = [];
let recording      = false;
let recordingStart = 0;
let analyser, animFrame;
let currentAudio   = null;   // currently playing Audio element

// ── DOM refs ───────────────────────────────────────────────
const messagesEl = document.getElementById('messages');
const textInput  = document.getElementById('textInput');
const sendBtn    = document.getElementById('sendBtn');
const voiceBtn   = document.getElementById('voiceBtn');
const waveform   = document.getElementById('waveform');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');


/* ═══════════════════════════════════════════════════════════
   WEBSOCKET
═══════════════════════════════════════════════════════════ */
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.type === 'status') setStatus(data.msg);
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}
connectWS();


/* ═══════════════════════════════════════════════════════════
   STATUS BAR
═══════════════════════════════════════════════════════════ */
const STATUS_MAP = {
  thinking:     ['thinking',  '🧠 Thinking…'],
  transcribing: ['thinking',  '📝 Transcribing…'],
  speaking:     ['speaking',  '🔊 Speaking…'],
  listening:    ['listening', '🎙️ Listening…'],
  done:         ['',          'Ready'],
  error:        ['error',     '⚠️ Error'],
};

function setStatus(key) {
  const [cls, label] = STATUS_MAP[key] || ['', key];
  statusDot.className = 'status-dot' + (cls ? ' ' + cls : '');
  statusText.textContent = label;
}


/* ═══════════════════════════════════════════════════════════
   INITIAL LOAD
═══════════════════════════════════════════════════════════ */
async function loadStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    document.getElementById('pillModel').textContent = d.model?.split('/').pop() || '—';
    document.getElementById('pillVoice').textContent = d.voice || '—';
    updateMemoryUI(d.memory || {});
  } catch (e) { /* server may still be starting */ }
}
loadStatus();


/* ═══════════════════════════════════════════════════════════
   MESSAGES
═══════════════════════════════════════════════════════════ */
function addMessage(role, content, label) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const lbl = role === 'user' ? 'You'
            : role === 'tool' ? (label || '🔧 Tool')
            : 'ARIA';
  wrap.innerHTML = `
    <div class="msg-label">${lbl}</div>
    <div class="msg-bubble">${escapeHtml(content)}</div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(t) {
  return t
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>')
    .replace(/```([^`]+)```/gs,
      (_, c) => `<code style="display:block;background:#111;padding:8px;border-radius:6px;
                  margin-top:6px;overflow-x:auto;font-size:12px">${c}</code>`);
}

// Welcome message on load
addMessage('aria', "Hello! I'm ARIA. Ask me anything — I can search the web, answer questions about your documents, or run Python code for you.");


/* ═══════════════════════════════════════════════════════════
   MEMORY / SIDEBAR UI
═══════════════════════════════════════════════════════════ */
function updateMemoryUI(mem) {
  const facts = mem.facts || {};

  // Facts panel
  const factsEl = document.getElementById('memoryFacts');
  if (Object.keys(facts).length === 0) {
    factsEl.textContent = 'No facts learned yet.';
  } else {
    factsEl.innerHTML = Object.entries(facts)
      .map(([k, v]) =>
        `<div class="memory-fact">
           <span class="fact-key">${k}</span>
           <span class="fact-val">${v}</span>
         </div>`)
      .join('');
  }

  // Stats
  document.getElementById('statDuration').textContent = mem.duration || '0s';
  document.getElementById('statTurns').textContent    = mem.turns    || 0;
  document.getElementById('statDocs').textContent     = (mem.docs || []).length;
  document.getElementById('pillTurns').textContent    = `${mem.turns || 0} turns`;

  // Docs list
  const docList = document.getElementById('docList');
  docList.innerHTML = (mem.docs || []).map(d => `
    <div class="doc-item">
      <span class="doc-name" title="${d}">📄 ${d}</span>
      <button class="doc-remove" onclick="removeDoc('${d}')" title="Remove">✕</button>
    </div>`).join('');
}


/* ═══════════════════════════════════════════════════════════
   TEXT CHAT
═══════════════════════════════════════════════════════════ */
textInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});

textInput.addEventListener('input', () => {
  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
});

sendBtn.addEventListener('click', sendText);

async function sendText() {
  const msg = textInput.value.trim();
  if (!msg) return;
  textInput.value = '';
  textInput.style.height = 'auto';
  addMessage('user', msg);
  sendBtn.disabled = true;
  setStatus('thinking');
  try {
    const r = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, speak: true }),
    });
    const d = await r.json();
    addMessage('aria', d.reply);
    updateMemoryUI(d.memory || {});
    if (d.audio_b64) playAudio(d.audio_b64);
    else setStatus('done');
  } catch (e) {
    setStatus('error');
  } finally {
    sendBtn.disabled = false;
  }
}


/* ═══════════════════════════════════════════════════════════
   VOICE RECORDING
═══════════════════════════════════════════════════════════ */
voiceBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
  // Interrupt ARIA if currently speaking
  if (currentAudio) {
    interruptSpeech();
    await startRecording();
    return;
  }
  if (!recording) await startRecording();
  else            await stopRecording();
}

async function startRecording() {
  if (recording) return;
  stopBargeInMonitor();   // free mic before opening a new stream
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
    });

    // Best supported format
    const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
      .find(m => MediaRecorder.isTypeSupported(m)) || '';

    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    audioChunks   = [];

    // Assign onstop BEFORE start() to avoid race condition
    mediaRecorder.onstop = async () => {
      const blob     = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      const duration = (Date.now() - recordingStart) / 1000;
      if (blob.size < 3000 || duration < 1.0) {
        setStatus('done');
        addMessage('aria', 'Recording too short — please speak for at least 1 second.');
        return;
      }
      await sendVoice(blob, mediaRecorder.mimeType);
    };

    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.start(250);
    recordingStart = Date.now();
    recording      = true;

    voiceBtn.classList.add('recording');
    voiceBtn.textContent = '⏹';
    voiceBtn.title       = 'Click to stop recording';
    setStatus('listening');
    startWaveform(stream);
  } catch (e) {
    alert('Microphone access denied. Please allow microphone permissions in your browser.');
  }
}

async function stopRecording() {
  if (!recording || !mediaRecorder) return;
  recording = false;
  voiceBtn.classList.remove('recording');
  voiceBtn.textContent = '🎤';
  voiceBtn.title       = 'Click to start recording';
  stopWaveform();
  mediaRecorder.requestData();                 // flush final chunk
  setTimeout(() => {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }, 150);
}

async function sendVoice(blob, mimeType) {
  setStatus('transcribing');
  const ext = (mimeType || '').includes('ogg') ? 'ogg' : 'webm';
  const fd  = new FormData();
  fd.append('audio', blob, `recording.${ext}`);
  try {
    const r = await fetch(`${API}/api/voice`, { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) {
      setStatus('error');
      addMessage('aria', `Voice error: ${d.error}`);
      return;
    }
    addMessage('user', d.transcript);
    addMessage('aria', d.reply);
    updateMemoryUI(d.memory || {});
    if (d.audio_b64) playAudio(d.audio_b64);
    else             setStatus('done');
  } catch (e) {
    setStatus('error');
    console.error('Voice send error:', e);
  }
}


/* ═══════════════════════════════════════════════════════════
   AUDIO PLAYBACK + BARGE-IN
═══════════════════════════════════════════════════════════ */
function playAudio(b64) {
  setStatus('speaking');
  const audio = new Audio('data:audio/mp3;base64,' + b64);
  currentAudio = audio;
  audio.play();
  audio.onended = () => {
    currentAudio = null;
    setStatus('done');
  };
  // NOTE: Auto barge-in via mic monitoring is disabled — speaker output
  // bleeds into the mic and causes false triggers on most setups.
  // Manual interrupt: click the 🎤 button at any time while ARIA speaks.
}

function interruptSpeech() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = '';
    currentAudio = null;
  }
  setStatus('done');
}

function stopBargeInMonitor() {
  // kept as a no-op so nothing else breaks
}


/* ═══════════════════════════════════════════════════════════
   WAVEFORM VISUALISER
═══════════════════════════════════════════════════════════ */
function startWaveform(stream) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  analyser  = ctx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);
  waveform.classList.add('visible');

  const canvasCtx = waveform.getContext('2d');
  const W = waveform.width  = waveform.offsetWidth;
  const H = waveform.height = 40;
  const buf = new Uint8Array(analyser.frequencyBinCount);

  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(buf);
    canvasCtx.clearRect(0, 0, W, H);
    const bars = 48;
    const bw   = W / bars;
    for (let i = 0; i < bars; i++) {
      const val   = buf[Math.floor(i * buf.length / bars)] / 255;
      const h     = Math.max(2, val * H);
      const alpha = 0.4 + val * 0.6;
      canvasCtx.fillStyle = `rgba(240,165,0,${alpha})`;
      canvasCtx.fillRect(i * bw + 1, (H - h) / 2, bw - 2, h);
    }
  }
  draw();
}

function stopWaveform() {
  cancelAnimationFrame(animFrame);
  waveform.classList.remove('visible');
  waveform.getContext('2d').clearRect(0, 0, waveform.width, waveform.height);
}


/* ═══════════════════════════════════════════════════════════
   DOCUMENT UPLOAD
═══════════════════════════════════════════════════════════ */
const fileInput  = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadZone.textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
    const d = await r.json();
    addMessage('tool', d.result, '📄 Document');
    updateMemoryUI({
      docs:  d.docs,
      turns: parseInt(document.getElementById('statTurns').textContent) || 0,
    });
  } catch (e) {
    addMessage('tool', 'Upload failed.', '⚠️ Error');
  } finally {
    uploadZone.innerHTML = '📎 Drop or click to upload<br/>PDF · TXT · MD · CSV · PY';
    fileInput.value = '';
  }
});

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.style.borderColor = 'var(--amber)'; });
uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor = ''; });
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
});

async function removeDoc(name) {
  await fetch(`${API}/api/docs/${encodeURIComponent(name)}`, { method: 'DELETE' });
  const r = await fetch(`${API}/api/status`);
  const d = await r.json();
  updateMemoryUI(d.memory || {});
}


/* ═══════════════════════════════════════════════════════════
   RESET
═══════════════════════════════════════════════════════════ */
document.getElementById('resetBtn').addEventListener('click', async () => {
  if (!confirm('Clear session memory?')) return;
  await fetch(`${API}/api/reset`, { method: 'POST' });
  messagesEl.innerHTML = '';
  addMessage('aria', 'Memory cleared. Fresh start!');
  updateMemoryUI({ facts: {}, turns: 0, duration: '0s', docs: [] });
  setStatus('done');
});