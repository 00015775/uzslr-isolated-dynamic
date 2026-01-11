import torch

# model settings
MODEL_PATH = "../modeling/notebooks/best_model.pth"  # use best_model.pth for inference
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