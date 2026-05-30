/**
 * ChristianAI — Modern AI Frontend
 */

'use strict';

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

const storedSessionId = sessionStorage.getItem('sessionId') || ('sess_' + Math.random().toString(36).slice(2, 11));
sessionStorage.setItem('sessionId', storedSessionId);
const storedHistory = JSON.parse(sessionStorage.getItem('messageHistory') || '[]');
const storedDenomination = sessionStorage.getItem('denomination') || 'General';

const state = {
  denomination: storedDenomination,
  isStreaming: false,
  isImageMode: false,
  sessionId: storedSessionId,
  messageHistory: storedHistory,
  verseCache: {},
};

function saveState() {
  sessionStorage.setItem('messageHistory', JSON.stringify(state.messageHistory));
  sessionStorage.setItem('denomination', state.denomination);
}

const dom = {
  denominationSelect: document.getElementById('denomination-select'),
  verificationBadge: document.getElementById('verification-badge'),
  messagesContainer: document.getElementById('messages-container'),
  welcomeScreen: document.getElementById('welcome-screen'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  charCount: document.getElementById('char-count'),
  suggestionList: document.getElementById('suggestion-list'),
  citationTooltip: document.getElementById('citation-tooltip'),
  imageToggleBtn: document.getElementById('image-toggle-btn'),
  newChatBtn: document.getElementById('new-chat-btn')
};

function init() {
  bindEvents();

  dom.denominationSelect.value = state.denomination;
  if (state.messageHistory.length > 0) {
    if (dom.welcomeScreen) dom.welcomeScreen.style.display = 'none';
    state.messageHistory.forEach(msg => {
      if (msg.role === 'user') {
        const text = msg.content.startsWith('/image ') ? msg.content.substring(7) : msg.content;
        appendUserMessage(text);
      } else if (msg.role === 'assistant') {
        const { bubbleEl } = appendAssistantMessage();
        if (msg.content.startsWith('[Image Generated:')) {
          const urlMatch = msg.content.match(/\[Image Generated: (.*?)\]/);
          if (urlMatch) {
            bubbleEl.innerHTML = `<img src="${urlMatch[1]}" style="max-width: 100%; max-height: 400px; object-fit: contain; border-radius: 8px; margin-top: 10px;" />`;
          } else {
            bubbleEl.innerHTML = renderResponseText(msg.content);
          }
        } else {
          bubbleEl.innerHTML = renderResponseText(msg.content);
        }
      }
    });
  }
}

function bindEvents() {
  dom.denominationSelect.addEventListener('change', (e) => {
    state.denomination = e.target.value;
    saveState();
  });

  dom.chatInput.addEventListener('input', onInputChange);
  dom.chatInput.addEventListener('keydown', onInputKeydown);
  dom.sendBtn.addEventListener('click', sendMessage);

  dom.imageToggleBtn.addEventListener('click', () => {
    state.isImageMode = !state.isImageMode;
    dom.imageToggleBtn.classList.toggle('active', state.isImageMode);
    dom.chatInput.placeholder = state.isImageMode ? "Describe the image you want to generate..." : "Ask anything...";
  });

  document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      dom.chatInput.value = btn.dataset.query;
      onInputChange();
      sendMessage();
    });
  });

  if (dom.newChatBtn) {
    dom.newChatBtn.addEventListener('click', () => {
      sessionStorage.removeItem('sessionId');
      sessionStorage.removeItem('messageHistory');
      window.location.reload();
    });
  }

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.citation-badge') && !e.target.closest('#citation-tooltip')) {
      dom.citationTooltip.classList.remove('visible');
    }
  });
}

function onInputChange() {
  const len = dom.chatInput.value.length;
  dom.charCount.textContent = `${len} / 2000`;
  dom.sendBtn.disabled = len === 0 || state.isStreaming;
  dom.chatInput.style.height = 'auto';
  dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 150) + 'px';
}

function onInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!dom.sendBtn.disabled) sendMessage();
  }
}

async function sendMessage() {
  const query = dom.chatInput.value.trim();
  if (!query || state.isStreaming) return;

  if (dom.welcomeScreen) dom.welcomeScreen.style.display = 'none';

  appendUserMessage(query); // show original text in UI
  dom.chatInput.value = '';
  onInputChange();

  if (state.isImageMode) {
    state.messageHistory.push({ role: 'user', content: `/image ${query}` });
    saveState();
    await generateImage(query);
  } else {
    state.messageHistory.push({ role: 'user', content: query });
    saveState();
    await streamAssistantResponse(query);
  }
}

function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `<div class="message-body"><div class="message-bubble">${escapeHtml(text)}</div></div>`;
  dom.messagesContainer.appendChild(el);
  scrollToBottom();
}

function appendAssistantMessage() {
  const messageEl = document.createElement('div');
  messageEl.className = 'message assistant';
  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'message-bubble';
  bubbleEl.innerHTML = '<span class="cursor"></span>';

  messageEl.innerHTML = `<div class="message-avatar">✝</div>`;
  const bodyEl = document.createElement('div');
  bodyEl.className = 'message-body';
  bodyEl.appendChild(bubbleEl);
  messageEl.appendChild(bodyEl);

  dom.messagesContainer.appendChild(messageEl);
  scrollToBottom();
  return { messageEl, bubbleEl };
}

async function generateImage(prompt) {
  state.isStreaming = true;
  dom.sendBtn.disabled = true;

  const { bubbleEl } = appendAssistantMessage();
  bubbleEl.innerHTML = `<div class="thinking-process" style="display: flex; align-items: center; gap: 8px; color: var(--primary-color); font-size: 0.95em; font-style: italic; font-weight: 500;">
    <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" stroke="currentColor">
      <style>.spinner_V8m1{transform-origin:center;animation:spinner_zKoa 2s linear infinite}@keyframes spinner_zKoa{100%{transform:rotate(360deg)}}</style>
      <circle cx="12" cy="12" r="9" fill="none" stroke-width="3" stroke-linecap="round" stroke-dasharray="15 41" class="spinner_V8m1" />
    </svg>
    <span class="thinking-text">Painting your vision (this may take 30-60 seconds)...</span>
  </div>`;

  try {
    const response = await fetch(`${API_BASE}/api/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, denomination: state.denomination }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    let imgHtml = `<img src="${data.url}" alt="Generated Image" style="max-width: 100%; max-height: 400px; object-fit: contain; border-radius: 8px; margin-top: 10px;" />`;
    if (data.safety_rewritten) {
      imgHtml += `<p style="font-size: 0.85em; color: var(--text-secondary); margin-top: 8px;"><em>Prompt adjusted for reverence/safety: "${data.revised_prompt}"</em></p>`;
    }

    bubbleEl.innerHTML = imgHtml;
    state.messageHistory.push({ role: 'assistant', content: `[Image Generated: ${data.url}]` });
    saveState();
  } catch (err) {
    bubbleEl.innerHTML = `<div class="safety-refusal">⚠️ ${err.message || 'Error generating image.'}</div>`;
  } finally {
    state.isStreaming = false;
    dom.sendBtn.disabled = false;
    scrollToBottom();
  }
}

async function streamAssistantResponse(query) {
  state.isStreaming = true;
  dom.sendBtn.disabled = true;
  dom.verificationBadge.style.display = 'block';
  updateVerificationBadge('checking');

  const { bubbleEl } = appendAssistantMessage();
  let currentText = '';

  bubbleEl.innerHTML = `<div class="thinking-process" style="color: var(--text-secondary); font-size: 0.9em; font-style: italic;">
    <span class="thinking-icon">⚙️</span> <span class="thinking-text">Evaluating input safety...</span>
  </div>`;

  let isThinking = true;

  try {
    const payload = {
      query,
      denomination: state.denomination,
      session_id: state.sessionId,
      history: state.messageHistory.slice(-10),
    };

    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          if (event.type === 'process' && isThinking) {
            const textEl = bubbleEl.querySelector('.thinking-text');
            if (textEl) textEl.textContent = event.data?.step || 'Processing...';
          } else if (event.type === 'token') {
            isThinking = false;
            currentText += event.data?.content || '';
            bubbleEl.innerHTML = renderResponseText(currentText) + '<span class="cursor"></span>';
            scrollToBottom();
          } else if (event.type === 'verification') {
            updateVerificationBadge(event.data?.passed ? 'passed' : 'failed');
          }
        } catch (e) { }
      }
    }

    bubbleEl.innerHTML = renderResponseText(currentText);
    state.messageHistory.push({ role: 'assistant', content: currentText });
    saveState();
  } catch (err) {
    bubbleEl.innerHTML = `<div class="safety-refusal">⚠️ An error occurred interacting with the server.</div>`;
    updateVerificationBadge('failed');
  } finally {
    state.isStreaming = false;
    dom.sendBtn.disabled = false;
  }
}

function updateVerificationBadge(status) {
  dom.verificationBadge.className = 'verification-badge';
  if (status === 'checking') {
    dom.verificationBadge.classList.add('state-checking');
    dom.verificationBadge.textContent = 'Verifying citations...';
  } else if (status === 'passed') {
    dom.verificationBadge.classList.add('state-passed');
    dom.verificationBadge.textContent = 'Citations Verified ✓';
  } else {
    dom.verificationBadge.classList.add('state-failed');
    dom.verificationBadge.textContent = 'Issues Detected';
  }
  setTimeout(() => { if (status !== 'checking') dom.verificationBadge.style.display = 'none'; }, 5000);
}

function renderResponseText(text) {
  let rendered = text.replace(/<thought_process>[\s\S]*?<\/thought_process>/g, '');
  rendered = rendered.replace(
    /\[([A-Za-z\s]+)\s+(\d+):(\d+(?:-\d+)?)\]/g,
    (match, book, chapter, verse) => {
      const ref = `${book.trim()} ${chapter}:${verse}`;
      return `<span class="citation-badge" onclick="showCitationTooltip(this, '${ref}')">${ref}</span>`;
    }
  );

  if (typeof marked !== 'undefined') {
    return marked.parse(rendered);
  }

  // Fallback
  rendered = rendered.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  return `<p>${rendered}</p>`;
}

window.showCitationTooltip = async function (el, ref) {
  const tooltip = dom.citationTooltip;
  if (!state.verseCache[ref]) {
    tooltip.innerHTML = `<em>Loading ${ref}...</em>`;
    state.verseCache[ref] = await fetchVerseText(ref);
  }
  tooltip.innerHTML = `<strong>${ref} (KJV)</strong><br><br>${state.verseCache[ref]}`;
  const rect = el.getBoundingClientRect();
  tooltip.style.left = `${Math.min(rect.left, window.innerWidth - 300)}px`;
  tooltip.style.top = `${rect.bottom + 8}px`;
  tooltip.classList.add('visible');
};

async function fetchVerseText(ref) {
  try {
    const match = ref.match(/^(.+)\s+(\d+):(\d+(?:-\d+)?)$/);
    if (!match) return 'Lookup failed.';
    const [, book, chapter, verse] = match;
    const response = await fetch(`${API_BASE}/api/verse?book=${encodeURIComponent(book)}&chapter=${chapter}&verse=${verse}`);
    if (response.ok) return (await response.json()).text;
    return 'Text not available.';
  } catch { return 'Connection failed.'; }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}
function scrollToBottom() {
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}

init();
