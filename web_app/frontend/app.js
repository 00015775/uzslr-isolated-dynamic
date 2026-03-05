// constants 
const BUFFER_SIZE = 32;
const WS_URL      = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/infer`;

const N_FACE = 468;
const N_POSE = 33;
const N_HAND = 21;

// DOM: camera / prediction 
const video      = document.getElementById('video');
const overlay    = document.getElementById('overlay');
const ctx        = overlay.getContext('2d');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const bufferBar  = document.getElementById('bufferBar');
const cameraWrap = document.getElementById('cameraWrapper');

const predictionIdle   = document.getElementById('predictionIdle');
const predictionResult = document.getElementById('predictionResult');
const predLabel        = document.getElementById('predLabel');
const confBar          = document.getElementById('confBar');
const confPct          = document.getElementById('confPct');
const viewSignLink     = document.getElementById('viewSignLink');
const historyList      = document.getElementById('historyList');

// DOM: LLM panel 
const llmPanel       = document.getElementById('llmPanel');
const llmToggle      = document.getElementById('llmToggle');
const llmControls    = document.getElementById('llmControls');
const llmModelSelect = document.getElementById('llmModelSelect');
const llmPrompt      = document.getElementById('llmPrompt');
const resetPromptBtn = document.getElementById('resetPromptBtn');
const clearSignsBtn  = document.getElementById('clearSignsBtn');
const collectedSigns = document.getElementById('collectedSigns');
const llmOutput      = document.getElementById('llmOutput');
const llmStatus      = document.getElementById('llmStatus');

// DOM: admin 
const rolePill           = document.getElementById('rolePill');
const roleDot            = document.getElementById('roleDot');
const roleLabel          = document.getElementById('roleLabel');
const roleDropdown       = document.getElementById('roleDropdown');
const switchToUser       = document.getElementById('switchToUser');
const switchToAdmin      = document.getElementById('switchToAdmin');
const adminModal         = document.getElementById('adminModal');
const adminModalClose    = document.getElementById('adminModalClose');
const adminPasswordInput = document.getElementById('adminPasswordInput');
const adminLoginBtn      = document.getElementById('adminLoginBtn');
const adminError         = document.getElementById('adminError');
const adminControls      = document.getElementById('adminControls');
const savePromptBtn      = document.getElementById('savePromptBtn');

// app state 
let ws                 = null;
let wsReady            = false;
let waitingForResponse = false;

// LLM config (from /api/config)
let llmEnabled          = false;
let llmHandAbsentFrames = 30;
let defaultSystemPrompt = '';

// Admin state
let isAdmin    = false;
let adminToken = null;

// CLIENT-SIDE STATE MACHINE 
// States:
//   IDLE          — no hands, nothing happening
//   FILLING       — hands visible, collecting 32 frames, bar animating
//   PREDICTED     — buffer just completed, prediction shown, bar full
//   ABSENT        — hands gone after prediction, counting down for LLM
//   LLM_FORMING   — POST in flight
//
// Transitions:
//   IDLE       + handsVisible=true              → FILLING (clear prediction, reset bar)
//   FILLING    + bufferFull=true + prediction   → PREDICTED (show prediction)
//   PREDICTED  + handsVisible=true              → FILLING (new sign: clear pred, reset bar)
//   PREDICTED  + handsVisible=false             → ABSENT (start absent counter)
//   ABSENT     + handsVisible=true              → FILLING (new sign)
//   ABSENT     + absentCount >= threshold       → LLM_FORMING (if LLM on) or IDLE
//   LLM_FORMING                                 → IDLE (after response)

const STATE = { IDLE: 0, FILLING: 1, PREDICTED: 2, ABSENT: 3, LLM_FORMING: 4 };
let appState     = STATE.IDLE;
let absentCount  = 0;

// Minimum ms to hold prediction visible before allowing next sign transition.
const PRED_HOLD_MS        = 4000;   // ms prediction card stays visible
let predictionLockedUntil = 0;   // timestamp — block state transitions until this

// LLM collection
let llmActive         = false;
let collectedSignsMap = new Map();  // sign → count
let lastCollectedSign = null;       // prevents adding same sign repeatedly

// state machine transition 
function transition(newState) {
  appState = newState;
}

function handleServerMessage(msg) {
  const handsVisible = msg.handsVisible;
  const bufferFull   = msg.bufferFull;
  const prediction   = msg.prediction;
  const confidence   = msg.confidence;
  const bufferSize   = msg.bufferSize;

  // always update buffer bar while filling
  if (appState === STATE.FILLING) {
    updateBufferBar(bufferSize);
  }

  switch (appState) {

    case STATE.IDLE:
      if (handsVisible) {
        transition(STATE.FILLING);
        clearPrediction();
        resetBufferBar();
        setStatus('collecting…', 'active');
        cameraWrap.classList.add('active');
        absentCount = 0;
      }
      break;

    case STATE.FILLING:
      if (!handsVisible) {
        // hands dropped before buffer filled — go back to idle
        transition(STATE.IDLE);
        resetBufferBar();
        setStatus('show both hands', 'inactive');
        cameraWrap.classList.remove('active');
        break;
      }
      if (bufferFull && prediction) {
        transition(STATE.PREDICTED);
        predictionLockedUntil = Date.now() + PRED_HOLD_MS;  // hold prediction visible
        updateBufferBar(BUFFER_SIZE);
        showPrediction(prediction, confidence);
        if (llmActive) maybeCollectSign(prediction);
        // reset bar after brief full-bar flash
        setTimeout(() => resetBufferBar(), 400);
      }
      break;

    case STATE.PREDICTED:
      // do not transition until prediction has been visible for PRED_HOLD_MS
      if (Date.now() < predictionLockedUntil) break;

      if (handsVisible) {
        // user started a new sign — clear prediction and start filling
        transition(STATE.FILLING);
        clearPrediction();
        resetBufferBar();
        setStatus('collecting…', 'active');
        absentCount = 0;
      } else {
        // hands gone — start absent countdown
        transition(STATE.ABSENT);
        absentCount = 1;
        updateAbsentStatus();
      }
      break;

    case STATE.ABSENT:
      if (handsVisible) {
        // hands came back — new sign
        transition(STATE.FILLING);
        clearPrediction();
        resetBufferBar();
        setStatus('collecting…', 'active');
        absentCount = 0;
        // if LLM was forming a sentence and hands return, clear old collection
        if (llmActive && !formingInProgress) {
          clearCollectionForNewSentence();
        }
        break;
      }
      absentCount++;
      updateAbsentStatus();
      if (llmActive && collectedSignsMap.size > 0 && !formingInProgress) {
        if (absentCount >= llmHandAbsentFrames) {
          transition(STATE.LLM_FORMING);
          formSentence();
        }
      } else if (absentCount > llmHandAbsentFrames + 10) {
        // no LLM or nothing collected — just go idle
        transition(STATE.IDLE);
        setStatus('show both hands', 'inactive');
        cameraWrap.classList.remove('active');
      }
      break;

    case STATE.LLM_FORMING:
      // waiting for POST response — ignore WebSocket messages except hands
      if (handsVisible) {
        // user started signing again before sentence finished
        // let formSentence() complete, then clearCollectionForNewSentence() on next FILLING
      }
      break;
  }
}

function updateAbsentStatus() {
  if (llmActive && collectedSignsMap.size > 0 && !formingInProgress) {
    const remaining = Math.max(0, llmHandAbsentFrames - absentCount);
    if (remaining > 0) {
      setStatus(`forming in ${remaining} frames…`, 'inactive');
    }
  } else {
    setStatus('show both hands', 'inactive');
    cameraWrap.classList.remove('active');
  }
}

// LLM sign collection 
function maybeCollectSign(sign) {
  if (sign === lastCollectedSign) return;  // same sign as last — skip
  lastCollectedSign = sign;
  if (collectedSignsMap.has(sign)) {
    collectedSignsMap.set(sign, collectedSignsMap.get(sign) + 1);
  } else {
    collectedSignsMap.set(sign, 1);
  }
  renderChips();
}

function clearCollectionForNewSentence() {
  collectedSignsMap.clear();
  lastCollectedSign = null;
  formingInProgress = false;
  renderChips();
  llmOutput.textContent = '';
  llmOutput.classList.remove('fade-out');
  setLlmStatus('');
}

let formingInProgress = false;

async function formSentence() {
  if (formingInProgress || collectedSignsMap.size === 0) return;
  formingInProgress = true;

  const model  = llmModelSelect.value;
  const prompt = llmPrompt.value;

  // send unique signs in order — the LLM prompt expects deduplicated words
  // e.g. Map{maktab:2, borish:1} → ['maktab', 'borish']  NOT ['maktab','maktab','borish']
  const signsList = [...collectedSignsMap.keys()];

  const shortName = model.split('/').pop();
  setLlmStatus(`Forming sentence with ${shortName}, please wait…`);
  setStatus('forming sentence…', 'inactive');

  try {
    const res  = await fetch('/api/form-sentence', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      // admin sends their current prompt text; regular users send empty -> server uses active prompt
    body:    JSON.stringify({ signs: signsList, systemPrompt: isAdmin ? prompt : '', model }),
    });
    const data = await res.json();

    llmOutput.textContent = data.sentence || '—';
    llmOutput.classList.remove('fade-out');
    setLlmStatus('');

    // fade out after 8 s then reset for next sentence
    setTimeout(() => {
      llmOutput.classList.add('fade-out');
      setTimeout(() => {
        clearCollectionForNewSentence();
        transition(STATE.IDLE);
        setStatus('show both hands', 'inactive');
        cameraWrap.classList.remove('active');
      }, 600);
    }, 15000);   // sentence stays visible for 15 s

  } catch (err) {
    setLlmStatus('Error forming sentence.');
    console.error('[llm]', err);
    formingInProgress = false;
    transition(STATE.IDLE);
  }
}

function resetCollection() {
  clearCollectionForNewSentence();   // also resets lastCollectedSign
  transition(STATE.IDLE);
  predictionLockedUntil = 0;         // clear any hold lock
  setStatus('show both hands', 'inactive');
  cameraWrap.classList.remove('active');
}

function renderChips() {
  collectedSigns.innerHTML = '';
  if (collectedSignsMap.size === 0) {
    collectedSigns.innerHTML = '<span class="collected-empty">No signs collected yet</span>';
    return;
  }
  collectedSignsMap.forEach((count, sign) => {
    const chip = document.createElement('span');
    chip.className = 'sign-chip';
    chip.innerHTML = `${sign.replace(/_/g,' ')}${count > 1 ? ` <span class="chip-count">×${count}</span>` : ''}`;
    collectedSigns.appendChild(chip);
  });
}

function setLlmStatus(msg) { llmStatus.textContent = msg; }

// LLM toggle & prompt controls 
function applyLlmToggle() {
  llmActive = llmToggle.checked;
  llmControls.classList.toggle('disabled', !llmActive);
  if (!llmActive) resetCollection();
}

llmToggle.addEventListener('change', applyLlmToggle);
clearSignsBtn.addEventListener('click', resetCollection);
resetPromptBtn.addEventListener('click', () => { llmPrompt.value = defaultSystemPrompt; });

document.querySelectorAll('input[name="promptMode"]').forEach(radio => {
  radio.addEventListener('change', () => {
    llmPrompt.readOnly = !(radio.value === 'edit' && llmActive);
  });
});

// config load 
async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    llmEnabled          = cfg.llmEnabled;
    llmHandAbsentFrames = cfg.llmHandAbsentFrames;
    defaultSystemPrompt = cfg.defaultSystemPrompt;

    if (llmEnabled) {
      cfg.llmModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        if (m === cfg.llmDefaultModel) opt.selected = true;
        llmModelSelect.appendChild(opt);
      });
      // use the server's currently active prompt (may differ from default if admin changed it)
      llmPrompt.value    = cfg.activePrompt || defaultSystemPrompt;
      llmPanel.style.display = 'block';
      applyLlmToggle();
    }
  } catch (e) { console.warn('[config]', e); }
}

// prediction UI
function clearPrediction() {
  predictionResult.classList.add('hidden');
  predictionIdle.classList.remove('hidden');
}

function showPrediction(sign, confidence) {
  const pct = Math.round(confidence * 100);
  predLabel.textContent = sign.replace(/_/g, ' ');
  confBar.style.width   = pct + '%';
  confPct.textContent   = pct + '%';
  viewSignLink.href     = `signs.html#${encodeURIComponent(sign)}`;
  predictionIdle.classList.add('hidden');
  predictionResult.classList.remove('hidden');
  predLabel.style.animation = 'none';
  predLabel.offsetHeight;
  predLabel.style.animation = '';
  addToHistory(sign, pct);
}

function addToHistory(sign, pct) {
  const first = historyList.firstElementChild;
  if (first && first.dataset.sign === sign) return;
  const li = document.createElement('li');
  li.dataset.sign = sign;
  li.innerHTML = `
    <span class="history-sign">${sign.replace(/_/g, ' ')}</span>
    <span class="history-conf">${pct}%</span>`;
  historyList.insertBefore(li, historyList.firstChild);
  while (historyList.children.length > 10) historyList.removeChild(historyList.lastChild);
}

// status / buffer bar
function setStatus(text, state = 'inactive') {
  statusText.textContent = text;
  statusDot.className    = `status-dot ${state}`;
}

function updateBufferBar(size) {
  bufferBar.style.width = Math.min(100, (size / BUFFER_SIZE) * 100) + '%';
}

function resetBufferBar() {
  bufferBar.style.width = '0%';
}

// WebSocket 
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen  = () => { wsReady = true; waitingForResponse = false; };
  ws.onclose = () => {
    wsReady = false; waitingForResponse = false;
    setStatus('disconnected', 'error');
    setTimeout(connectWS, 2000);
  };
  ws.onerror = () => {
    wsReady = false; waitingForResponse = false;
    setStatus('connection error', 'error');
  };
  ws.onmessage = (e) => {
    waitingForResponse = false;
    handleServerMessage(JSON.parse(e.data));
  };
}

// landmark vector 
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
];

function buildLandmarkVector(results) {
  const vec = new Float32Array(1662);
  let o = 0;
  if (results.faceLandmarks)      for (const l of results.faceLandmarks)      { vec[o++]=l.x; vec[o++]=l.y; vec[o++]=l.z; }
  else                            o += N_FACE * 3;
  if (results.poseLandmarks)      for (const l of results.poseLandmarks)      { vec[o++]=l.x; vec[o++]=l.y; vec[o++]=l.z; vec[o++]=l.visibility??0; }
  else                            o += N_POSE * 4;
  if (results.rightHandLandmarks) for (const l of results.rightHandLandmarks) { vec[o++]=l.x; vec[o++]=l.y; vec[o++]=l.z; }
  else                            o += N_HAND * 3;
  if (results.leftHandLandmarks)  for (const l of results.leftHandLandmarks)  { vec[o++]=l.x; vec[o++]=l.y; vec[o++]=l.z; }
  else                            o += N_HAND * 3;
  return vec;
}

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (results.rightHandLandmarks) { drawConn(results.rightHandLandmarks,'#c8f135',2); drawDots(results.rightHandLandmarks,'#c8f135',3); }
  if (results.leftHandLandmarks)  { drawConn(results.leftHandLandmarks, '#3de8c8',2); drawDots(results.leftHandLandmarks, '#3de8c8',3); }
  if (results.faceLandmarks)      drawDots(results.faceLandmarks,'rgba(255,255,255,0.1)',1);
}

function drawConn(lms, color, lw) {
  ctx.strokeStyle=color; ctx.lineWidth=lw;
  for (const [a,b] of HAND_CONNECTIONS) {
    if (!lms[a]||!lms[b]) continue;
    ctx.beginPath();
    ctx.moveTo(lms[a].x*overlay.width, lms[a].y*overlay.height);
    ctx.lineTo(lms[b].x*overlay.width, lms[b].y*overlay.height);
    ctx.stroke();
  }
}
function drawDots(lms,color,r) {
  ctx.fillStyle=color;
  for (const l of lms) { ctx.beginPath(); ctx.arc(l.x*overlay.width,l.y*overlay.height,r,0,Math.PI*2); ctx.fill(); }
}

// MediaPipe 
function initMediaPipe() {
  const holistic = new Holistic({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1635989137/${f}`
  });
  holistic.setOptions({
    modelComplexity: 0, smoothLandmarks: true,
    enableSegmentation: false, smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  holistic.onResults(results => {
    overlay.width  = video.videoWidth  || overlay.clientWidth;
    overlay.height = video.videoHeight || overlay.clientHeight;
    drawResults(results);
    if (!wsReady || waitingForResponse) return;
    const hasBothHands = results.rightHandLandmarks != null && results.leftHandLandmarks != null;
    waitingForResponse = true;
    ws.send(JSON.stringify({ landmarks: Array.from(buildLandmarkVector(results)), hasBothHands }));
  });
  navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
    .then(stream => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        new Camera(video, { onFrame: async () => { await holistic.send({ image: video }); }, width:640, height:480 }).start();
        setStatus('show both hands', 'inactive');
      };
    })
    .catch(err => { console.error('[camera]', err); setStatus('camera access denied','error'); });
}

// boot 
// admin logic 

// role pill dropdown 
let dropdownOpen = false;

function openDropdown()  { dropdownOpen = true;  roleDropdown.classList.remove('hidden'); }
function closeDropdown() { dropdownOpen = false; roleDropdown.classList.add('hidden'); }

rolePill.addEventListener('click', e => {
  e.stopPropagation();
  dropdownOpen ? closeDropdown() : openDropdown();
});

// close when clicking anywhere outside
document.addEventListener('click', e => {
  if (dropdownOpen && !roleDropdown.contains(e.target)) closeDropdown();
});

// close on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeDropdown();
    closeAdminModal();
  }
});

switchToUser.addEventListener('click', () => {
  closeDropdown();
  if (isAdmin) exitAdminMode();
});

switchToAdmin.addEventListener('click', () => {
  closeDropdown();
  if (!isAdmin) openAdminModal();
});

// modal
function openAdminModal() {
  adminPasswordInput.value = '';
  adminError.style.display = 'none';
  adminModal.classList.add('open');
  setTimeout(() => adminPasswordInput.focus(), 80);
}
function closeAdminModal() {
  adminModal.classList.remove('open');
  adminPasswordInput.value = '';
  adminError.style.display = 'none';
}

async function attemptAdminLogin() {
  const pw = adminPasswordInput.value.trim();
  if (!pw) return;
  adminLoginBtn.disabled = true;
  adminError.style.display = 'none';
  try {
    const res = await fetch('/api/admin/verify', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ password: pw }),
    });
    if (!res.ok) {
      adminError.style.display = 'block';
      adminPasswordInput.value = '';
      adminPasswordInput.classList.add('shake');
      setTimeout(() => adminPasswordInput.classList.remove('shake'), 400);
      return;
    }
    const data = await res.json();
    adminToken = data.token;
    isAdmin    = true;
    closeAdminModal();
    enterAdminMode();
  } catch(e) {
    adminError.style.display = 'block';
  } finally {
    adminLoginBtn.disabled = false;
  }
}

function enterAdminMode() {
  roleDot.className  = 'role-dot admin';
  roleLabel.textContent = 'Admin';
  adminControls.classList.remove('hidden');
}

function exitAdminMode() {
  isAdmin = false; adminToken = null;
  roleDot.className  = 'role-dot user';
  roleLabel.textContent = 'User';
  adminControls.classList.add('hidden');
  document.querySelector('input[name="promptMode"][value="readonly"]').checked = true;
  llmPrompt.readOnly = true;
}

async function savePrompt() {
  if (!isAdmin || !adminToken) return;
  const prompt = llmPrompt.value.trim();
  if (!prompt) return;
  savePromptBtn.disabled = true;
  savePromptBtn.textContent = 'Saving…';
  try {
    const res = await fetch('/api/admin/prompt', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prompt, token: adminToken }),
    });
    savePromptBtn.textContent = res.ok ? 'Saved ✓' : 'Error — try again';
  } catch(e) {
    savePromptBtn.textContent = 'Error — try again';
  } finally {
    savePromptBtn.disabled = false;
    setTimeout(() => { savePromptBtn.textContent = 'Save prompt for all users'; }, 2000);
  }
}

adminModalClose.addEventListener('click', closeAdminModal);
adminModal.addEventListener('click', e => { if (e.target === adminModal) closeAdminModal(); });
savePromptBtn.addEventListener('click', savePrompt);
adminLoginBtn.addEventListener('click', attemptAdminLogin);
adminPasswordInput.addEventListener('keydown', e => { if (e.key === 'Enter') attemptAdminLogin(); });

// init
setStatus('starting…', 'inactive');
loadConfig();
connectWS();
initMediaPipe();
