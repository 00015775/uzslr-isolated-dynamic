import json
import sys
import pathlib
import numpy as np
import torch
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# path resolution
# this file lives at: <repo>/web_app/backend/main.py
_BACKEND_DIR = pathlib.Path(__file__).parent.resolve()   # .../web_app/backend
_WEBAPP_DIR  = _BACKEND_DIR.parent                        # .../web_app
_REPO_ROOT   = _WEBAPP_DIR.parent                         # .../uzslr-isolated-dynamic

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web_app.backend.config import (
    MAX_LEN, CHANNELS, NUM_CLASSES, MODEL_DIM,
    BUFFER_SIZE, HAND_DISAPPEAR_TOLERANCE, DEFAULT_SIGNS, get_device,
    LLM_ENABLED, LLM_MODELS, LLM_DEFAULT_MODEL,
    LLM_HAND_ABSENT_FRAMES, DEFAULT_SYSTEM_PROMPT,
)
from web_app.backend.preprocess import Preprocess
from web_app.backend.model import SignLanguageModel

# absolute paths to assets
_MODEL_PATH   = _WEBAPP_DIR / "best_model.pth"
_SIGNS_DIR    = _REPO_ROOT  / "show-50-signs" / "signs"
_FRONTEND_DIR = _WEBAPP_DIR / "frontend"


# global singletons
device     = None
model      = None
preprocess = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model, preprocess

    # sign recognition model
    device = get_device()
    print(f"[startup] using device: {device}")

    model = SignLanguageModel(
        max_len=MAX_LEN, channels=CHANNELS,
        num_classes=NUM_CLASSES, dim=MODEL_DIM,
    )
    model.load_state_dict(
        torch.load(str(_MODEL_PATH), map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"[startup] sign model loaded from {_MODEL_PATH}")

    preprocess = Preprocess(max_len=MAX_LEN).to(device)

    # ollama (only in LLM image)
    if LLM_ENABLED:
        from web_app.backend.llm_client import start_ollama
        start_ollama()

    print("[startup] ready")
    yield

    # shutdown
    if LLM_ENABLED:
        from web_app.backend.llm_client import stop_ollama
        stop_ollama()
    print("[shutdown] done")


app = FastAPI(lifespan=lifespan)


# REST: frontend config
@app.get("/api/config")
async def get_config():
    """
    Called by the frontend on load to know whether LLM features are available.
    Keeps all feature-flag logic server-side.
    """
    return JSONResponse({
        "llmEnabled":          LLM_ENABLED,
        "llmModels":           LLM_MODELS,
        "llmDefaultModel":     LLM_DEFAULT_MODEL,
        "llmHandAbsentFrames": LLM_HAND_ABSENT_FRAMES,
        "defaultSystemPrompt": DEFAULT_SYSTEM_PROMPT,
    })


# REST: sentence formation
class SentenceRequest(BaseModel):
    signs:        list[str]
    systemPrompt: str
    model:        str = LLM_DEFAULT_MODEL


@app.post("/api/form-sentence")
async def form_sentence_endpoint(req: SentenceRequest):
    if not LLM_ENABLED:
        raise HTTPException(status_code=503, detail="LLM not available in this image")

    if not req.signs:
        raise HTTPException(status_code=400, detail="No signs provided")

    from web_app.backend.llm_client import form_sentence
    try:
        sentence = await form_sentence(
            signs=req.signs,
            system_prompt=req.systemPrompt,
            model=req.model,
        )
        return JSONResponse({"sentence": sentence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket: real-time sign inference
@app.websocket("/ws/infer")
async def websocket_infer(ws: WebSocket):
    await ws.accept()

    frame_buffer           = []          # plain list — we control clearing manually
    hand_disappear_counter = 0
    hands_were_visible     = False       # latched: were hands visible last cycle?

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            landmarks:      list = msg["landmarks"]
            has_both_hands: bool = msg["hasBothHands"]

            vec = np.array(landmarks, dtype=np.float32)

            # hand tracking
            if has_both_hands:
                hand_disappear_counter = 0
                if not hands_were_visible:
                    # hands just appeared — start a fresh buffer for a new sign
                    frame_buffer       = []
                    hands_were_visible = True
            else:
                if hands_were_visible:
                    hand_disappear_counter += 1
                    if hand_disappear_counter > HAND_DISAPPEAR_TOLERANCE:
                        hands_were_visible = False
                        frame_buffer       = []

            # buffer & predict
            prediction = None
            confidence = None
            buffer_full = False

            if hands_were_visible and len(frame_buffer) < BUFFER_SIZE:
                frame_buffer.append(vec)

            if hands_were_visible and len(frame_buffer) == BUFFER_SIZE:
                buffer_full = True
                frames = np.stack(frame_buffer)
                x      = torch.from_numpy(frames).to(device)
                x      = preprocess(x)
                x      = x.unsqueeze(0)
                mask   = torch.ones(1, MAX_LEN, dtype=torch.bool, device=device)

                with torch.no_grad():
                    logits               = model(x, mask)
                    probs                = torch.softmax(logits, dim=-1)
                    conf, pred_idx       = torch.max(probs, dim=-1)

                prediction = DEFAULT_SIGNS[pred_idx.item()]
                confidence = round(conf.item(), 3)

                # clear buffer so next sign gets a fresh 32 frames
                frame_buffer = []

            response = {
                "handsVisible":  hands_were_visible,
                "bufferSize":    len(frame_buffer),
                "bufferFull":    buffer_full,        # client uses this to reset bar
                "prediction":    prediction,
                "confidence":    confidence,
            }

            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass


# static files
app.mount("/signs", StaticFiles(directory=str(_SIGNS_DIR)),    name="signs")
app.mount("/",      StaticFiles(directory=str(_FRONTEND_DIR),  html=True), name="frontend")
