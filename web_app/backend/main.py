import json
import sys
import pathlib
import numpy as np
import torch
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# path resolution
# this file lives at:  <repo>/web-app/backend/main.py
_BACKEND_DIR = pathlib.Path(__file__).parent.resolve()   # .../web-app/backend
_WEBAPP_DIR  = _BACKEND_DIR.parent                        # .../web-app
_REPO_ROOT   = _WEBAPP_DIR.parent                         # .../uzslr-isolated-dynamic

# make sure the repo root is on sys.path so "web_app.backend.*" imports work
# regardless of where uvicorn is launched from
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web_app.backend.config import (
    MAX_LEN, CHANNELS, NUM_CLASSES, MODEL_DIM,
    BUFFER_SIZE, HAND_DISAPPEAR_TOLERANCE, DEFAULT_SIGNS, get_device
)
from web_app.backend.preprocess import Preprocess
from web_app.backend.model import SignLanguageModel

# absolute paths to assets
_MODEL_PATH   = _WEBAPP_DIR / "best_model.pth"
_SIGNS_DIR    = _REPO_ROOT  / "show-50-signs" / "signs"
_FRONTEND_DIR = _WEBAPP_DIR / "frontend"


# global model (loaded once at startup)
device = None
model = None
preprocess = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model, preprocess
    device = get_device()
    print(f"[startup] using device: {device}")

    model = SignLanguageModel(
        max_len=MAX_LEN,
        channels=CHANNELS,
        num_classes=NUM_CLASSES,
        dim=MODEL_DIM,
    )
    model.load_state_dict(
        torch.load(str(_MODEL_PATH), map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"[startup] model loaded from {_MODEL_PATH}")

    preprocess = Preprocess(max_len=MAX_LEN).to(device)
    print("[startup] ready")
    yield
    print("[shutdown] done")


app = FastAPI(lifespan=lifespan)


# WebSocket inference endpoint
@app.websocket("/ws/infer")
async def websocket_infer(ws: WebSocket):
    await ws.accept()

    frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
    hand_disappear_counter = 0
    inference_active = False
    both_hands_visible = False

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            landmarks: list = msg["landmarks"]       # list[float] length 1662
            has_both_hands: bool = msg["hasBothHands"]

            vec = np.array(landmarks, dtype=np.float32)

            # hand-state machine (mirrors inference04_main.py)
            if has_both_hands:
                both_hands_visible = True
                hand_disappear_counter = 0
                if not inference_active:
                    inference_active = True
                    frame_buffer.clear()
            else:
                if both_hands_visible:
                    hand_disappear_counter += 1
                    if hand_disappear_counter > HAND_DISAPPEAR_TOLERANCE:
                        both_hands_visible = False
                        inference_active = False
                        frame_buffer.clear()

            # buffer + predict
            response: dict = {
                "inferenceActive": inference_active,
                "bufferSize": len(frame_buffer),
                "prediction": None,
                "confidence": None,
            }

            if inference_active:
                frame_buffer.append(vec)

                if len(frame_buffer) == BUFFER_SIZE:
                    frames = np.stack(list(frame_buffer))          # (32, 1662)
                    x = torch.from_numpy(frames).to(device)
                    x = preprocess(x)                              # (32, 708)
                    x = x.unsqueeze(0)                             # (1, 32, 708)
                    mask = torch.ones(1, MAX_LEN, dtype=torch.bool, device=device)

                    with torch.no_grad():
                        logits = model(x, mask)
                        probs = torch.softmax(logits, dim=-1)
                        confidence, pred_idx = torch.max(probs, dim=-1)

                    response["prediction"] = DEFAULT_SIGNS[pred_idx.item()]
                    response["confidence"] = round(confidence.item(), 3)

            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass


# static files (frontend)
app.mount("/signs", StaticFiles(directory=str(_SIGNS_DIR)), name="signs")
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")