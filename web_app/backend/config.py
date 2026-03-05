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
    "kmamaroziqov/alloma-3b-q4",
    "kmamaroziqov/alloma-1b-q4",
]
LLM_DEFAULT_MODEL = "kmamaroziqov/alloma-3b-q4"

# N frames hands must be absent before sentence formation triggers.
# ~30 frames ≈ 2-3 s at real-world MediaPipe throughput on a laptop/server.
LLM_HAND_ABSENT_FRAMES = 30

DEFAULT_SYSTEM_PROMPT = """Berilgan so\u2018zlardan bitta o\u2018zbek jumlasi yoz.

TAQIQLANGAN:
- "Men ... deb jumla yaratishim mumkin" kabi iboralar
- "Men bu vazifani bajara olmayman" yoki rad etish
- Izoh, tushuntirish, qavslar yoki qo\u2018shimcha gap
- Raqamlar (1. 2. va h.k.)
- So\u2018zlardan birini takrorlash yoki faqat bitta so\u2018z yozish

RUXSAT:
- Grammatik qo\u2018shimchalar qo\u2018shish (ga, da, ni, bilan, va, uchun, dan)
- So\u2018zlar bog\u2018liq bo\u2018lmasa ham eng mantiqiy jumlani tuzish

FORMAT: Faqat bitta tugallangan jumla. Boshqa hech narsa.

Misollar:

So\u2018zlar: assalomu alaykum, iltimos, yordam berish
Javob: Assalomu alaykum, iltimos menga yordam bering.

So\u2018zlar: maktab, futbol, o\u2018ynash
Javob: Maktabda futbol o\u2018ynayapmiz.

So\u2018zlar: bozor, kartoshka, pomidor, kichik
Javob: Bozordan kichik kartoshka va pomidor oldim.

So\u2018zlar: maktab, kitob, qidirish
Javob: Maktabda kitob qidiryapman.

So\u2018zlar: qish, qor, yomg\u2019ir
Javob: Qishda qor va yomg\u2019ir yog\u2019di.

So\u2018zlar: restoran, birga, musiqa
Javob: Restoranda birga musiqa tingladik.

So\u2018zlar: bozor, yomg\u2019ir, yopish
Javob: Yomg\u2019ir tufayli bozor yopildi.

So\u2018zlar: quyon, kichik, o\u2018ynash
Javob: Kichik quyon o\u2018ynayapti.

So\u2018zlar: qor, yomg\u2019ir, jismoniy tarbiya
Javob: Qor va yomg\u2019irli havoda jismoniy tarbiya qildik.

So\u2018zlar: assalomu alaykum, yopish, maktab, yomg\u2019ir
Javob: Assalomu alaykum, yomg\u2019ir tufayli maktab yopildi.

So\u2018zlar: tosh, shokolad, metro
Javob: Metroda tosh va shokolad ko\u2018rdim.

Endi:"""


# Admin mode unlocks model selector + prompt editor in the UI.
ADMIN_PASSWORD = "uzslr-admin-2024"

