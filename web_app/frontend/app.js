// constants
const BUFFER_SIZE = 32;
const WS_URL      = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/infer`;

// landmark counts matching Python
const N_FACE = 468;
const N_POSE = 33;
const N_HAND = 21;

// DOM refs
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

// state
let ws          = null;
let wsReady     = false;
let currentSign = null;

// Back-pressure: only send next frame once server responds to previous.
// This prevents a growing queue of unprocessed messages causing lag.
let waitingForResponse = false;

// websocket
function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsReady = true;
    waitingForResponse = false;
    console.log('[ws] connected');
  };

  ws.onmessage = (e) => {
    waitingForResponse = false; // unblock next send immediately on response
    const msg = JSON.parse(e.data);
    updateBufferBar(msg.bufferSize);
    updateStatus(msg.inferenceActive);
    if (msg.prediction !== null) {
      showPrediction(msg.prediction, msg.confidence);
    }
  };

  ws.onclose = () => {
    wsReady = false;
    waitingForResponse = false;
    setStatus('disconnected', 'error');
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => {
    wsReady = false;
    waitingForResponse = false;
    setStatus('connection error', 'error');
  };
}

// status helpers
function setStatus(text, state = 'inactive') {
  statusText.textContent = text;
  statusDot.className = `status-dot ${state}`;
}

function updateStatus(inferenceActive) {
  if (inferenceActive) {
    setStatus('recognising…', 'active');
    cameraWrap.classList.add('active');
  } else {
    setStatus('show both hands', 'inactive');
    cameraWrap.classList.remove('active');
  }
}

function updateBufferBar(size) {
  bufferBar.style.width = Math.min(100, (size / BUFFER_SIZE) * 100) + '%';
}

// prediction UI
function showPrediction(sign, confidence) {
  const pct = Math.round(confidence * 100);
  currentSign = sign;

  const display = sign.replace(/_/g, ' ');
  predLabel.textContent = display;
  confBar.style.width   = pct + '%';
  confPct.textContent   = pct + '%';
  viewSignLink.href     = `signs.html#${encodeURIComponent(sign)}`;

  predictionIdle.classList.add('hidden');
  predictionResult.classList.remove('hidden');

  // replay pop animation
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
    <span class="history-conf">${pct}%</span>
  `;
  historyList.insertBefore(li, historyList.firstChild);
  while (historyList.children.length > 10) {
    historyList.removeChild(historyList.lastChild);
  }
}

// landmark vector builder
// Mirrors inference04_main.py extract_landmarks()
// Output: Float32Array of length 1662
//   [face(468*3), pose(33*4), rightHand(21*3), leftHand(21*3)]
function buildLandmarkVector(results) {
  const vec = new Float32Array(1662);
  let offset = 0;

  if (results.faceLandmarks) {
    for (const lm of results.faceLandmarks) {
      vec[offset++] = lm.x; vec[offset++] = lm.y; vec[offset++] = lm.z;
    }
  } else { offset += N_FACE * 3; }

  if (results.poseLandmarks) {
    for (const lm of results.poseLandmarks) {
      vec[offset++] = lm.x; vec[offset++] = lm.y;
      vec[offset++] = lm.z; vec[offset++] = lm.visibility ?? 0;
    }
  } else { offset += N_POSE * 4; }

  if (results.rightHandLandmarks) {
    for (const lm of results.rightHandLandmarks) {
      vec[offset++] = lm.x; vec[offset++] = lm.y; vec[offset++] = lm.z;
    }
  } else { offset += N_HAND * 3; }

  if (results.leftHandLandmarks) {
    for (const lm of results.leftHandLandmarks) {
      vec[offset++] = lm.x; vec[offset++] = lm.y; vec[offset++] = lm.z;
    }
  } else { offset += N_HAND * 3; }

  return vec;
}

// drawing
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (results.rightHandLandmarks) {
    drawConnections(results.rightHandLandmarks, '#c8f135', 2);
    drawDots(results.rightHandLandmarks, '#c8f135', 3);
  }
  if (results.leftHandLandmarks) {
    drawConnections(results.leftHandLandmarks, '#3de8c8', 2);
    drawDots(results.leftHandLandmarks, '#3de8c8', 3);
  }
  if (results.faceLandmarks) {
    drawDots(results.faceLandmarks, 'rgba(255,255,255,0.1)', 1);
  }
}

function drawConnections(landmarks, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  for (const [a, b] of HAND_CONNECTIONS) {
    const lmA = landmarks[a], lmB = landmarks[b];
    if (!lmA || !lmB) continue;
    ctx.beginPath();
    ctx.moveTo(lmA.x * overlay.width, lmA.y * overlay.height);
    ctx.lineTo(lmB.x * overlay.width, lmB.y * overlay.height);
    ctx.stroke();
  }
}

function drawDots(landmarks, color, r) {
  ctx.fillStyle = color;
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * overlay.width, lm.y * overlay.height, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

// mediapipe holistic
function initMediaPipe() {
  const holistic = new Holistic({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1635989137/${file}`
  });

  holistic.setOptions({
    modelComplexity: 0,          // 0 = lite, ~2× faster than 1 on CPU
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  holistic.onResults((results) => {
    overlay.width  = video.videoWidth  || overlay.clientWidth;
    overlay.height = video.videoHeight || overlay.clientHeight;

    drawResults(results);

    // skip send if WS not ready or still awaiting previous response
    if (!wsReady || waitingForResponse) return;

    const landmarks    = buildLandmarkVector(results);
    const hasBothHands = results.rightHandLandmarks != null &&
                         results.leftHandLandmarks  != null;

    waitingForResponse = true;
    ws.send(JSON.stringify({
      landmarks: Array.from(landmarks),
      hasBothHands,
    }));
  });

  // 640×480 is plenty for landmark extraction and ~2× faster to process
  navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
    .then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        const camera = new Camera(video, {
          onFrame: async () => { await holistic.send({ image: video }); },
          width: 640,
          height: 480,
        });
        camera.start();
        setStatus('show both hands', 'inactive');
      };
    })
    .catch((err) => {
      console.error('[camera] error:', err);
      setStatus('camera access denied', 'error');
    });
}

// boot
setStatus('starting…', 'inactive');
connectWS();
initMediaPipe();
