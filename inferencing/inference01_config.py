import torch

# model settings
MODEL_PATH = "best_model.pth"  # use best_model.pth for inference
MAX_LEN = 32
CHANNELS = 708
NUM_CLASSES = 50
MODEL_DIM = 192  # or 384

# camera settings
VIDEO_DEVICE = 1  # change if needed
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# mediapipe settings
MP_CONFIDENCE = 0.5

# inference settings
BUFFER_SIZE = 32  # must match MAX_LEN
HANDS_REQUIRED = True  # require both hands visible to start inference
HAND_DISAPPEAR_TOLERANCE = 5  # allow hands to disappear for 5 frames

# sign labels, must match training order
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']

# device selection
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

# LLM settings
# required libraries
# pip install mlx-lm

# now download the load 
# python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-1.5B-Instruct-4bit')"
# link for model in HF: https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit

USE_LOCAL_LLM = False # Set to false to use API LLMs
LLM_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 50
LLM_TEMPERATURE = 0.7 # more for creativity

# Sentence buffer settings
MIN_SIGNS_FOR_SENTENCE = 2
MAX_SIGNS_PER_SENTENCE = 15
SIGN_STABILITY_THRESHOLD = 2
SIGN_CONFIDENCE_THRESHOLD = 0.6

# System prompt in Uzbek

SYSTEM_PROMPT = """Siz berilgan so'zlardan to'g'ri va mantiqiy jumla tuzasiz.

Qoidalar:
- So'zlarni grammatik jihatdan to'g'ri bog'lang
- Mantiqli va tushunarli jumla yarating
- Faqat yakuniy jumlani yozing, boshqa hech narsa qo'shmang

Misollar:

So'zlar: men, maktab, borish
Jumla: Men maktabga boraman.

So'zlar: kitob, qayerda, kutubxona
Jumla: Kitob kutubxonada.

So'zlar: ovqat, tayyorlash, oshxona
Jumla: Ovqatni oshxonada tayyorlayman.

So'zlar: iltimos, yordam, kerak
Jumla: Iltimos, menga yordam kerak.

Endi jumla tuzing:"""



# SYSTEM_PROMPT = """Siz o'zbek imo-ishora tilini o'zbek tiliga tarjimon siz. 
# Berilgan imo-ishora so'zlaridan to'g'ri grammatik jumla tuzing.
# Faqat yakuniy jumlani yozing, boshqa hech narsa qo'shmang.

# Misollar:

# Imo-ishora: assalomu_alaykum, ism, nima
# Jumla: Assalomu alaykum, ismingiz nima?

# Imo-ishora: men, maktab, borish, xohlash
# Jumla: Men maktabga borishni xohlayman.

# Imo-ishora: kitob, qayerda
# Jumla: Kitob qayerda?

# Endi tarjima qiling:"""


