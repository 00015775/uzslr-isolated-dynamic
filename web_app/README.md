# UzSLR — Uzbek Sign Language Recognition

A real-time web application for recognising **50 isolated Uzbek signs** using a webcam. An optional LLM mode collects recognised signs and attempts to form grammatically correct Uzbek sentences using a locally-running language model.

> Docker Image: https://hub.docker.com/repository/docker/00015775/uzslr-web

---

## Demo

| Sign recognition | LLM sentence formation |
|---|---|
| Show both hands to the camera → the model processes 32 frames and predicts the sign | Sign multiple words → hide hands → a sentence is formed automatically |

---

## Features

- **Real-time recognition** of 50 Uzbek signs via webcam
- **MediaPipe Holistic** landmark extraction running entirely in the browser
- **PyTorch model** for sign classification
- **50 Signs gallery** with reference GIFs for every sign, searchable and with fullscreen view
- **LLM Mode** *(optional)* — collects signs and forms Uzbek sentences using [alloma-1b-q4](https://ollama.com/kmamaroziqov/alloma-1b-q4) running locally via Ollama
- Fully **offline** after first load — no external API calls, no data sent anywhere

---

## Quick start — Docker (recommended)

### Without LLM (sign recognition only)

```bash
docker pull 00015775/uzslr-web:1.0.0 
docker run -p 7860:7860 00015775/uzslr-web:1.0.0 
```

Open [http://localhost:7860](http://localhost:7860)

### With LLM (sign recognition + sentence formation)

```bash
docker pull 00015775/uzslr-web:2.0.0 

# on cpu
docker run -p 7860:7860 00015775/uzslr-web:2.0.0 

# on gpu
docker run --gpus all -p 7860:7860 00015775/uzslr-web:2.0.0 
```

Open [http://localhost:7860](http://localhost:7860) — the **LLM Mode** toggle will be visible in the right panel.

> **Note:** The LLM image is ~7 GB due to the bundled Uzbek language model. First pull may take a few minutes.

---

## Quick start — Local Python (no Docker)

### Prerequisites

- Python 3.9
- [Ollama](https://ollama.com) installed *(only needed for LLM mode)*

### 1. Clone and install

```bash
git clone https://github.com/00015775/uzslr-isolated-dynamic.git

cd uzslr-isolated-dynamic
# create and activate the environment
conda env create -f .web_app/environment-web-uzslr-signs.yml

conda activate web-uzslr-signs
```

### 2. Run — sign recognition only

```bash
# from root directory
uvicorn web_app.backend.main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### 3. Run — with LLM mode enabled

In one terminal, start Ollama:
```bash
# if ollama not available locally, then download
curl -fsSL https://ollama.com/install.sh | sh

# pull llm model
ollama pull kmamaroziqov/alloma-1b-q4   # one-time download (~807 MB)
ollama pull kmamaroziqov/alloma-3b-q4   # one-time download (~2 GB)

ollama serve
```

In another terminal:
```bash
# from root directory
LLM_ENABLED=true uvicorn web_app.backend.main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the **LLM Mode** toggle will appear.

---

## How it works

```
Browser                          Server (FastAPI)
───────                          ────────────────
Webcam frames
  → MediaPipe Holistic           
  → 1662-dim landmark vector  →  WebSocket /ws/infer
                                   → preprocessing (centering, normalisation,
                                                    velocity, acceleration)
                                   → PyTorch model (32 frames → 1 prediction)
                               ←  predicted sign + confidence

[LLM mode only]
Collected signs list         →  POST /api/form-sentence
                                   → Ollama (alloma-1b-q4)
                             ←  formed Uzbek sentence
```

### Recognised signs (50 total)

`assalomu_alaykum` · `bahor` · `birga` · `bo'sh` · `bosh_kiyim` · `boshlanishi` · `bozor` · `eshik` · `futbol` · `iltimos` · `internet` · `javob` · `jismoniy_tarbiya` · `karam` · `kartoshka` · `kichik` · `kitob` · `ko'prik` · `likopcha` · `maktab` · `mehmonxona` · `mehribon` · `metro` · `musiqa` · `o'simlik_yog'i` · `o'ynash` · `ochish` · `ot` · `ovqat_tayyorlash` · `oxiri` · `poezd` · `pomidor` · `qidirish` · `qish` · `qo'ziqorin` · `qor` · `qorong'i` · `quyon` · `restoran` · `sariyog'` · `shokolad` · `sovun` · `stakan` · `televizor` · `tosh` · `toza` · `turish` · `yomg'ir` · `yopish` · `yordam_berish`

---

## Project structure

```
uzslr-isolated-dynamic/
├── web_app/
│   ├── backend/
│   │   ├── main.py          # FastAPI app — WebSocket inference + REST endpoints
│   │   ├── config.py        # Model config, LLM settings, sign labels
│   │   ├── preprocess.py    # Landmark preprocessing pipeline
│   │   ├── model.py         # PyTorch transformer model definition
│   │   └── llm_client.py    # Ollama subprocess management + sentence formation
│   ├── frontend/
│   │   ├── index.html       # Main recognition page
│   │   ├── signs.html       # 50 Signs gallery with fullscreen modal
│   │   ├── app.js           # MediaPipe + WebSocket + LLM UI logic
│   │   └── style.css        # Dark industrial UI theme
│   ├── best_model.pth       # Trained sign recognition model weights
│   ├── Dockerfile           # Base image — sign recognition only
│   ├── Dockerfile.llm       # LLM image — includes Ollama + alloma-1b-q4
│   ├── requirements.txt     # Python dependencies
│   ├── ollama-models/       # .gitignored, model lives here
│   └── README.md
...
```

---

## Building Docker images yourself

### Base image (no LLM)

```bash
# from repo root
docker build --platform=linux/amd64 -f web_app/Dockerfile -t uzslr:latest .
```

### LLM image

```bash
# copy Ollama model cache (must have pulled the model first)
mkdir -p web_app/ollama-models
cp -r ~/.ollama/models/blobs     web_app/ollama-models/blobs
cp -r ~/.ollama/models/manifests web_app/ollama-models/manifests

docker build --platform=linux/amd64 -f web_app/Dockerfile.llm -t uzslr:llm .
```

> **Apple Silicon (M1/M2/M3):** The `--platform=linux/amd64` flag is required when building for deployment on amd64 servers. Docker Desktop handles the cross-compilation via Rosetta.

---

## Browser requirements

> [!NOTE]
> LLM inference speed depends on the host machine, and processing 32 frames depends on the camera’s FPS.

Any modern browser with WebRTC support: Chrome, Firefox, Safari, Edge. Camera permission required. Works on desktop and mobile.
