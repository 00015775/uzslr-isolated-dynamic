"""
llm_client.py
Manages the Ollama subprocess lifecycle and exposes a single async function
`form_sentence(signs, system_prompt, model)` that returns an Uzbek sentence.

The alloma model requires:
  1. Uzbek apostrophe normalisation: replace ʻ ʼ ' ' etc -> APST before sending,
     then replace APST -> ' in the output.  (Mirrors the HuggingFace reference
     implementation at kmamaroziqov/alloma-1b-q4.)
  2. Manual chat-template formatting using the model's special tokens rather
     than relying on Ollama's /api/chat endpoint to apply it automatically.
     Using /api/generate with a pre-formatted prompt is more reliable.
"""

import logging
import os
import re
import subprocess
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_BIN  = os.environ.get("OLLAMA_BIN", "/usr/local/bin/ollama")

# All apostrophe-like characters the model cannot handle natively
_APST_PATTERN = re.compile(r"[''‚‛ʻʼʽʾʿˈˊˋˌˍ'\']")

_ollama_process = None  # type: Optional[subprocess.Popen]


# apostrophe helpers
def _encode_apostrophes(text: str) -> str:
    """Replace Uzbek apostrophes with the APST placeholder."""
    return _APST_PATTERN.sub("APST", text)


def _decode_apostrophes(text: str) -> str:
    """Restore APST placeholder back to a clean apostrophe."""
    return text.replace("APST", "'")


# chat template 
# Matches the alloma model's expected format exactly (Llama-3 style).
_BOS   = "<|begin_of_text|>"
_EOT   = "<|eot_id|>"
_SYS_OPEN  = "<|start_header_id|>system<|end_header_id|>\n"
_USR_OPEN  = "<|start_header_id|>user<|end_header_id|>\n"
_ASST_OPEN = "<|start_header_id|>assistant<|end_header_id|>"

def _build_prompt(system: str, user: str) -> str:
    """
    Build a fully-formatted prompt string for /api/generate.

    The alloma model was fine-tuned with a minimal system turn ("You are a
    helpful assistant"). It barely attends to a long system prompt, so we:
      - Keep the system turn short and generic (like the HF reference)
      - Move ALL rules and examples into the user turn
      - Apostrophe-encode both turns
    """
    # generic system turn — matches HF fine-tuning style
    sys_content = _encode_apostrophes("You are a helpful assistant")

    # user turn = full instruction block + examples + actual word list
    # system arg (our Uzbek prompt) goes here where the model actually reads it
    user_content = _encode_apostrophes(system + "\n\n" + user)

    return (
        f"{_BOS}{_SYS_OPEN}{sys_content}{_EOT}"
        f"{_USR_OPEN}{user_content}{_EOT}"
        f"{_ASST_OPEN}"
    )


# ollama process management 
def start_ollama() -> None:
    """Launch Ollama server as a background subprocess."""
    global _ollama_process
    if _ollama_process is not None and _ollama_process.poll() is None:
        logger.info("[ollama] already running")
        return

    logger.info(f"[ollama] starting server from {OLLAMA_BIN}")
    _ollama_process = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # wait until the API is reachable (max 30 s)
    for _ in range(60):
        try:
            import urllib.request
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1)
            logger.info("[ollama] server ready")
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError("[ollama] server did not become ready in 30 s")


def stop_ollama() -> None:
    """Terminate the Ollama subprocess on shutdown."""
    global _ollama_process
    if _ollama_process and _ollama_process.poll() is None:
        logger.info("[ollama] stopping server")
        _ollama_process.terminate()
        try:
            _ollama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ollama_process.kill()
    _ollama_process = None


# main inference call
async def form_sentence(
    signs: list,
    system_prompt: str,
    model: str = "kmamaroziqov/alloma-3b-q4",
) -> str:
    """
    Send collected sign words to the alloma model and return a formed sentence.

    signs         : deduplicated list, e.g. ['maktab', 'borish']
    system_prompt : Uzbek system prompt from the UI
    model         : ollama model name
    """
    # Issue 2 fix: clean underscores before sending to model
    # e.g. 'yordam_berish' -> 'yordam berish', 'o\'ynash' stays as-is (no underscore)
    clean_signs  = [s.replace('_', ' ') for s in signs]
    words_line   = ", ".join(clean_signs)
    user_message = f"So'zlar: {words_line}"

    # build fully-formatted raw prompt with apostrophe encoding
    raw_prompt = _build_prompt(system_prompt, user_message)

    payload = {
        "model":  model,
        "prompt": raw_prompt,
        "stream": False,
        "raw":    True,          # tell Ollama NOT to apply its own template
        "options": {
            "temperature": 0.3,       # low = consistent, structured output (3B handles this fine)
            "num_predict": 60,        # slightly more room for longer natural sentences
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
            "stop": [_EOT, "\n"],
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",   # /generate not /chat
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    raw_output = data.get("response", "").strip()

    # decode APST back to apostrophes and remove special tokens
    sentence = _decode_apostrophes(raw_output)
    sentence = sentence.replace(_EOT, "").strip()

    # strip "Jumla:" or "Javob:" prefix the model sometimes adds
    for prefix in ("Jumla:", "jumla:", "Javob:", "javob:"):
        if sentence.lower().startswith(prefix.lower()):
            sentence = sentence[len(prefix):].strip()

    # strip parenthetical commentary: "Sentence. (explanation...)"
    sentence = re.sub(r"\s*\([^)]*\)\s*$", "", sentence).strip()
    # strip square bracket notes: "Sentence. [note]"
    sentence = re.sub(r"\s*\[[^\]]*\]\s*$", "", sentence).strip()
    # keep only the first sentence if model generated multiple
    parts = re.split(r"(?<=[.!?])\s+", sentence)
    sentence = parts[0].strip() if parts else sentence

    return sentence