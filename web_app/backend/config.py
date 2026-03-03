import torch
import os

# model settings
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")
MAX_LEN = 32
CHANNELS = 708
NUM_CLASSES = 50
MODEL_DIM = 192  # base model

# mediapipe settings
MP_CONFIDENCE = 0.5

# inference settings
BUFFER_SIZE = 32  # must match MAX_LEN
HAND_DISAPPEAR_TOLERANCE = 5

# sign labels — must match training order
DEFAULT_SIGNS = [
    'assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi',
    'bozor', 'eshik', 'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya',
    'karam', 'kartoshka', 'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab',
    'mehmonxona', 'mehribon', 'metro', 'musiqa', "o'simlik_yog'i", "o'ynash",
    'ochish', 'ot', 'ovqat_tayyorlash', 'oxiri', 'poezd', 'pomidor', 'qidirish',
    'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 'restoran', "sariyog'",
    'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza', 'turish',
    "yomg'ir", 'yopish', 'yordam_berish'
]

def get_device():
    # MPS not used in server context (not stable for inference serving)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# LLM settings
# LLM_ENABLED is set to "true" only in Dockerfile.llm via ENV.
# In the base image it is absent, so all LLM UI is hidden/disabled.
LLM_ENABLED = os.environ.get("LLM_ENABLED", "false").lower() == "true"

LLM_MODELS = [
    "kmamaroziqov/alloma-1b-q4",
]
LLM_DEFAULT_MODEL = "kmamaroziqov/alloma-1b-q4"

# N frames hands must be absent before sentence formation triggers.
# ~30 frames ≈ 2-3 s at real-world MediaPipe throughput on a laptop/server.
LLM_HAND_ABSENT_FRAMES = 30

DEFAULT_SYSTEM_PROMPT = """Siz berilgan so\u2019zlardan faqat BITTA qisqa va to\u2018g\u2018ri o\u2018zbek jumlasi tuzasiz.

Qat\u2019iy qoidalar:
- Faqat BITTA jumla yozing
- Hech qanday izoh, qavslar yoki qo\u2018shimcha tushuntirish qo\u2018shmang
- Qavs ichida hech narsa yozmang
- So\u2018zlarni grammatik jihatdan to\u2018g\u2018ri bog\u2018lang
- Faqat yakuniy jumlani yozing, boshqa hech narsa yo\u2018q

Misollar:

So\u2018zlar: men, maktab, borish
Javob: Men maktabga boraman.

So\u2018zlar: kitob, kutubxona
Javob: Kitob kutubxonada.

So\u2018zlar: ovqat, tayyorlash, oshxona
Javob: Ovqatni oshxonada tayyorlayman.

So\u2018zlar: quyon, assalomu alaykum, jismoniy tarbiya
Javob: Quyon bilan salomlashib, jismoniy tarbiya qilamiz.

So\u2018zlar: iltimos, yordam, kerak
Javob: Iltimos, menga yordam kerak.

Endi javob bering:"""
