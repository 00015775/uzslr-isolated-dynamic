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
